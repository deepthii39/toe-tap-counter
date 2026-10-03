import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

// ================================================================
// MODEL CONSTANTS
// ================================================================

// Both models MUST be exported with this same imgsz (see export_models.py).
// 320 = ~4x fewer pixels to preprocess and ~4x less network compute than 640.
// Must be a multiple of 32 (320, 384, 416, 480, 640).
const int kInputSize = 320;

const int kGrid8 = kInputSize ~/ 8;
const int kGrid16 = kInputSize ~/ 16;
const int kGrid32 = kInputSize ~/ 32;
const int kCandidates = kGrid8 * kGrid8 + kGrid16 * kGrid16 + kGrid32 * kGrid32;

const int kPoseChannels = 56; // 4 box + 1 score + 17 * 3 keypoints
const int kKeypointStart = 5;

// COCO keypoints, in the order returned from the worker:
// [L ankle, R ankle, L knee, R knee].
const List<int> kKeypointIds = [15, 16, 13, 14];

const String kPoseAsset = 'assets/yolo26n-pose.tflite';
const String kBallAsset = 'assets/football.tflite';

// Class index of the ball in the ball model.
//   custom single-class model -> 0
//   plain COCO detect model   -> 32 ("sports ball")
const int kBallClass = 32;

const double kPersonThreshold = 0.25;
const double kAnkleThreshold = 0.30;
const double kKneeThreshold = 0.30;

// Ball: a normal detection needs kBallThreshold. If we already track a ball,
// a weaker detection near its last position is accepted (down to kBallLow).
const double kBallThreshold = 0.25;
const double kBallLow = 0.12;

// ---- Speed settings ----
// Measured on a phone: preprocessing ~5 ms, pose ~250 ms, ball ~195 ms,
// so the two models are what costs time.
//
// kDelegate:
//   'none'    = plain CPU interpreter (known to work; default)
//   'xnnpack' = explicit XNNPACK CPU delegate (try after 'none' runs fine)
//   'gpu'     = GPU delegate (can be much faster, but may fail or crash on
//               some phones; needs OpenCL)
const String kDelegate = 'none';
const int kThreads = 4; // try 2 or 3 if it gets slower (big.LITTLE CPUs)

// Run the ball model on every Nth frame (1 = every frame). The ball moves
// much less than the legs, so its last position is reused in between.
const int kBallEvery = 2;

InterpreterOptions _makeOptions() {
  final options = InterpreterOptions()..threads = kThreads;
  try {
    if (kDelegate == 'xnnpack') {
      options.addDelegate(
        XNNPackDelegate(
          options: XNNPackDelegateOptions(numThreads: kThreads),
        ),
      );
    } else if (kDelegate == 'gpu') {
      options.addDelegate(GpuDelegateV2());
    }
  } catch (e) {
    print('Delegate "$kDelegate" unavailable ($e); using plain CPU.');
  }
  return options;
}

// Result layout (Float64List, length 21):
// [0] person score  [1] upright w  [2] upright h  [3] worker ms
// [4 + i*3 .. 6 + i*3] = x, y, conf for keypoint i
//     i: 0 = L ankle, 1 = R ankle, 2 = L knee, 3 = R knee
// [16] ball score  [17] ball cx  [18] ball cy  [19] ball w  [20] ball h
const int kResultLength = 21;

// ================================================================
// MAIN
// ================================================================

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  final cameras = await availableCameras();

  if (cameras.isEmpty) {
    runApp(
      const MaterialApp(
        home: Scaffold(
          body: Center(child: Text('No camera found')),
        ),
      ),
    );
    return;
  }

  final camera = cameras.firstWhere(
    (c) => c.lensDirection == CameraLensDirection.back,
    orElse: () => cameras.first,
  );

  runApp(ToeTapApp(camera: camera));
}

class ToeTapApp extends StatelessWidget {
  const ToeTapApp({super.key, required this.camera});

  final CameraDescription camera;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Toe Tap',
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: ToeTapPage(camera: camera),
    );
  }
}

// ================================================================
// BACKGROUND ISOLATE
// ================================================================
//
// Camera frame -> rotate upright -> letterbox -> RGB tensor (ONCE)
//   -> pose model  -> 4 keypoints
//   -> ball model  -> ball box (guided by last known ball position)
// All coordinates returned are in UPRIGHT image pixels.

int _frameCounter = 0;

class _Letterbox {
  _Letterbox(this.scale, this.padX, this.padY, this.uw, this.uh);

  final double scale;
  final double padX;
  final double padY;
  final double uw;
  final double uh;

  double x(double mx) => ((mx - padX) / scale).clamp(0.0, uw).toDouble();
  double y(double my) => ((my - padY) / scale).clamp(0.0, uh).toDouble();
}

bool? _nchwOf(List<int> shape) {
  final j = shape.join(',');
  if (j == '1,3,$kInputSize,$kInputSize') return true;
  if (j == '1,$kInputSize,$kInputSize,3') return false;
  return null;
}

void _workerMain(SendPort toMain) {
  final port = ReceivePort();
  toMain.send(port.sendPort);

  Interpreter? pose;
  Interpreter? ball;
  Float32List? input;
  bool nchw = true;

  port.listen((dynamic message) {
    final map = message as Map;
    final type = map['type'];

    if (type == 'init') {
      try {
        final poseBytes = (map['pose'] as TransferableTypedData)
            .materialize()
            .asUint8List();

        final p = Interpreter.fromBuffer(
          poseBytes,
          options: _makeOptions(),
        );
        p.allocateTensors();

        final pin = p.getInputTensor(0).shape;
        final pout = p.getOutputTensor(0).shape;
        print('POSE  in=$pin out=$pout');

        final layout = _nchwOf(pin);
        if (layout == null) {
          throw StateError('Unexpected pose input: $pin (kInputSize = '
              '$kInputSize). Re-export the pose model with imgsz=$kInputSize.');
        }
        if (pout.join(',') != '1,$kPoseChannels,$kCandidates') {
          throw StateError('Unexpected pose output: $pout');
        }
        nchw = layout;

        Interpreter? b;
        final ballMsg = map['ball'];
        if (ballMsg != null) {
          final ballBytes =
              (ballMsg as TransferableTypedData).materialize().asUint8List();
          b = Interpreter.fromBuffer(
            ballBytes,
            options: _makeOptions(),
          );
          b.allocateTensors();

          final bin = b.getInputTensor(0).shape;
          final bout = b.getOutputTensor(0).shape;
          print('BALL  in=$bin out=$bout');

          if (_nchwOf(bin) != nchw) {
            throw StateError('Ball model input $bin does not match pose '
                'model input $pin (same imgsz + layout required)');
          }
          if (bout.length != 3 ||
              bout[0] != 1 ||
              bout[2] != kCandidates ||
              bout[1] <= 4 + kBallClass) {
            throw StateError('Unexpected ball output: $bout. Export with '
                'end2end=False so the shape is [1, 4+classes, $kCandidates].');
          }
        }

        pose = p;
        ball = b;
        input = Float32List(3 * kInputSize * kInputSize);

        toMain.send({'type': 'ready', 'ball': b != null});
      } catch (e) {
        toMain.send({'type': 'error', 'message': '$e'});
      }
      return;
    }

    if (type == 'frame') {
      final p = pose;
      final buf = input;

      if (p == null || buf == null) {
        toMain.send({'type': 'frame_error'});
        return;
      }

      try {
        final result = _processFrame(p, ball, buf, nchw, map);
        toMain.send({'type': 'result', 'data': result});
      } catch (e, st) {
        print('Worker frame error: $e\n$st');
        toMain.send({'type': 'frame_error'});
      }
    }
  });
}

Float32List _run(Interpreter it, Float32List input) {
  it.getInputTensor(0).data = input.buffer.asUint8List();
  it.invoke();
  final b = it.getOutputTensor(0).data;
  return Float32List.view(b.buffer, b.offsetInBytes, b.lengthInBytes ~/ 4);
}

Float64List _processFrame(
  Interpreter pose,
  Interpreter? ball,
  Float32List input,
  bool nchw,
  Map map,
) {
  final swTotal = Stopwatch()..start();

  final srcW = map['w'] as int;
  final srcH = map['h'] as int;
  final rotation = map['rotation'] as int;
  final yStride = map['yStride'] as int;
  final uvStride = map['uvStride'] as int;
  final uvPixel = map['uvPixel'] as int;

  // Last known ball (upright px). hr <= 0 means "no hint".
  final hx = (map['hx'] as num).toDouble();
  final hy = (map['hy'] as num).toDouble();
  final hr = (map['hr'] as num).toDouble();

  final yBytes =
      (map['y'] as TransferableTypedData).materialize().asUint8List();
  final uBytes =
      (map['u'] as TransferableTypedData).materialize().asUint8List();
  final vBytes =
      (map['v'] as TransferableTypedData).materialize().asUint8List();

  final swap = rotation == 90 || rotation == 270;
  final uw = swap ? srcH : srcW;
  final uh = swap ? srcW : srcH;

  final scale = math.min(kInputSize / uw, kInputSize / uh);
  final contentW = math.max(1, (uw * scale).round());
  final contentH = math.max(1, (uh * scale).round());
  final startX = ((kInputSize - contentW) / 2).round();
  final startY = ((kInputSize - contentH) / 2).round();
  final endX = math.min(kInputSize, startX + contentW);
  final endY = math.min(kInputSize, startY + contentH);

  final uxTable = Int32List(kInputSize);
  final uyTable = Int32List(kInputSize);

  for (int m = 0; m < kInputSize; m++) {
    uxTable[m] = (((m - startX) + 0.5) / scale).floor().clamp(0, uw - 1);
    uyTable[m] = (((m - startY) + 0.5) / scale).floor().clamp(0, uh - 1);
  }

  // Upright -> sensor pixel is linear:
  //   rx = rxC + rxX*ux + rxY*uy     ry = ryC + ryX*ux + ryY*uy
  // so there is no switch inside the pixel loop.
  int rxC, rxX, rxY, ryC, ryX, ryY;
  switch (rotation) {
    case 90:
      rxC = 0;
      rxX = 0;
      rxY = 1;
      ryC = srcH - 1;
      ryX = -1;
      ryY = 0;
      break;
    case 180:
      rxC = srcW - 1;
      rxX = -1;
      rxY = 0;
      ryC = srcH - 1;
      ryX = 0;
      ryY = -1;
      break;
    case 270:
      rxC = srcW - 1;
      rxX = 0;
      rxY = -1;
      ryC = 0;
      ryX = 1;
      ryY = 0;
      break;
    default:
      rxC = 0;
      rxX = 1;
      rxY = 0;
      ryC = 0;
      ryX = 0;
      ryY = 1;
  }

  input.fillRange(0, input.length, 114.0 / 255.0);

  final plane = kInputSize * kInputSize;
  const inv255 = 1.0 / 255.0;

  for (int my = startY; my < endY; my++) {
    final uy = uyTable[my];
    final rxRow = rxC + rxY * uy;
    final ryRow = ryC + ryY * uy;
    final rowBase = my * kInputSize;

    for (int mx = startX; mx < endX; mx++) {
      final ux = uxTable[mx];
      final rx = rxRow + rxX * ux;
      final ry = ryRow + ryX * ux;

      final yVal = yBytes[ry * yStride + rx];
      final uvIndex = (ry >> 1) * uvStride + (rx >> 1) * uvPixel;
      final u = uBytes[uvIndex] - 128;
      final v = vBytes[uvIndex] - 128;

      // Integer YUV -> RGB (BT.601, coefficients scaled by 256).
      int r = yVal + ((359 * v) >> 8);
      int g = yVal - ((88 * u + 183 * v) >> 8);
      int b = yVal + ((454 * u) >> 8);

      r = r < 0 ? 0 : (r > 255 ? 255 : r);
      g = g < 0 ? 0 : (g > 255 ? 255 : g);
      b = b < 0 ? 0 : (b > 255 ? 255 : b);

      final idx = rowBase + mx;

      if (nchw) {
        input[idx] = r * inv255;
        input[plane + idx] = g * inv255;
        input[2 * plane + idx] = b * inv255;
      } else {
        final base = idx * 3;
        input[base] = r * inv255;
        input[base + 1] = g * inv255;
        input[base + 2] = b * inv255;
      }
    }
  }

  final prepMs = swTotal.elapsedMicroseconds / 1000.0;

  final lb = _Letterbox(
    scale,
    startX.toDouble(),
    startY.toDouble(),
    uw.toDouble(),
    uh.toDouble(),
  );

  final res = Float64List(kResultLength);
  res[1] = uw.toDouble();
  res[2] = uh.toDouble();

  final swPose = Stopwatch()..start();
  _decodePose(_run(pose, input), res, lb);
  swPose.stop();

  final swBall = Stopwatch()..start();
  if (ball != null && _frameCounter % kBallEvery == 0) {
    _decodeBall(_run(ball, input), res, lb, hx, hy, hr);
  }
  swBall.stop();

  swTotal.stop();
  res[3] = swTotal.elapsedMicroseconds / 1000.0;

  if (_frameCounter++ % 30 == 0) {
    print('timing prep=${prepMs.toStringAsFixed(0)}ms '
        'pose=${(swPose.elapsedMicroseconds / 1000).toStringAsFixed(0)}ms '
        'ball=${(swBall.elapsedMicroseconds / 1000).toStringAsFixed(0)}ms '
        'total=${res[3].toStringAsFixed(0)}ms');
  }

  return res;
}

// ---------------------------------------------------------------
// POSE DECODE
// ---------------------------------------------------------------
// Different Ultralytics exports store keypoints differently:
//   mode 0: absolute pixels in the model input
//   mode 1: normalised 0..1 (typical for TFLite exports)
//   mode 2: raw offsets that still need (raw + anchor) * stride
// We try all three and keep the one whose points fall inside the
// detected person box. The chosen mode is printed in the log.

void _decodePose(Float32List out, Float64List res, _Letterbox lb) {
  int best = -1;
  double bestScore = 0.0;
  const scoreOffset = 4 * kCandidates;

  for (int c = 0; c < kCandidates; c++) {
    final s = out[scoreOffset + c];
    if (s > bestScore) {
      bestScore = s;
      best = c;
    }
  }

  res[0] = bestScore;
  if (best < 0 || bestScore < kPersonThreshold) return;

  // Person box (same units as the keypoints).
  final rawW = out[2 * kCandidates + best];
  final unit = rawW <= 2.0 ? kInputSize.toDouble() : 1.0;
  final bcx = out[best] * unit;
  final bcy = out[kCandidates + best] * unit;
  final bw = rawW * unit;
  final bh = out[3 * kCandidates + best] * unit;

  // Box grown by 30% per side so feet at the edge still count as inside.
  final x0 = bcx - bw * 0.8;
  final x1 = bcx + bw * 0.8;
  final y0 = bcy - bh * 0.8;
  final y1 = bcy + bh * 0.8;

  // Anchor + stride of the winning candidate.
  const n8 = kGrid8 * kGrid8;
  const n16 = kGrid16 * kGrid16;
  double ax;
  double ay;
  double stride;

  if (best < n8) {
    ax = (best % kGrid8) + 0.5;
    ay = (best ~/ kGrid8) + 0.5;
    stride = 8.0;
  } else if (best < n8 + n16) {
    final l = best - n8;
    ax = (l % kGrid16) + 0.5;
    ay = (l ~/ kGrid16) + 0.5;
    stride = 16.0;
  } else {
    final l = best - n8 - n16;
    ax = (l % kGrid32) + 0.5;
    ay = (l ~/ kGrid32) + 0.5;
    stride = 32.0;
  }

  final n = kKeypointIds.length;
  final raw = List<double>.filled(n * 3, 0.0);

  for (int i = 0; i < n; i++) {
    final base = (kKeypointStart + kKeypointIds[i] * 3) * kCandidates;
    raw[i * 3] = out[base + best];
    raw[i * 3 + 1] = out[base + kCandidates + best];
    raw[i * 3 + 2] = out[base + 2 * kCandidates + best];
  }

  (double, double) toModel(int mode, double rx, double ry) {
    switch (mode) {
      case 0:
        return (rx, ry);
      case 1:
        return (rx * kInputSize, ry * kInputSize);
      default:
        return ((rx + ax) * stride, (ry + ay) * stride);
    }
  }

  int mode = 0;
  int bestHits = -1;
  for (int m = 0; m < 3; m++) {
    int hits = 0;
    for (int i = 0; i < n; i++) {
      final (mx, my) = toModel(m, raw[i * 3], raw[i * 3 + 1]);
      if (mx >= x0 && mx <= x1 && my >= y0 && my <= y1) hits++;
    }
    // strictly greater: ties keep the earlier (more plausible) mode
    if (hits > bestHits) {
      bestHits = hits;
      mode = m;
    }
  }

  for (int i = 0; i < n; i++) {
    final (mx, my) = toModel(mode, raw[i * 3], raw[i * 3 + 1]);
    res[4 + i * 3] = lb.x(mx);
    res[5 + i * 3] = lb.y(my);
    res[6 + i * 3] = raw[i * 3 + 2].clamp(0.0, 1.0).toDouble();
  }

  if (_frameCounter % 30 == 0) {
    print('pose person=${bestScore.toStringAsFixed(2)} mode=$mode '
        'hits=$bestHits');
  }
}

// ---------------------------------------------------------------
// BALL DECODE  (output [1, 4 + classes, N], boxes are cx,cy,w,h)
// ---------------------------------------------------------------
// Guided by the last known ball position (hx, hy, hr):
//  - candidates near it get a ranking bonus (avoids jumping to a
//    random round object elsewhere in the frame)
//  - near it, weak detections (>= kBallLow) are accepted, which keeps
//    the track alive while the foot partly covers the ball.

void _decodeBall(
  Float32List out,
  Float64List res,
  _Letterbox lb,
  double hx,
  double hy,
  double hr,
) {
  final scoreOffset = (4 + kBallClass) * kCandidates;
  final hasHint = hr > 0;
  final gate = hasHint ? hr * 4.0 + 20.0 : 0.0;

  int best = -1;
  double bestRank = 0.0;
  double bestScore = 0.0;
  double bcx = 0, bcy = 0, bw = 0, bh = 0;

  for (int c = 0; c < kCandidates; c++) {
    final s = out[scoreOffset + c];
    if (s < kBallLow) continue;
    if (s < kBallThreshold && !hasHint) continue;

    double cx = out[c];
    double cy = out[kCandidates + c];
    double w = out[2 * kCandidates + c];
    double h = out[3 * kCandidates + c];

    // TFLite exports usually give normalised boxes.
    if (w <= 2.0 && h <= 2.0) {
      const k = kInputSize + 0.0;
      cx *= k;
      cy *= k;
      w *= k;
      h *= k;
    }

    final ux = lb.x(cx);
    final uy = lb.y(cy);

    double rank = s;
    if (hasHint) {
      final dx = ux - hx;
      final dy = uy - hy;
      final near = math.sqrt(dx * dx + dy * dy) <= gate;
      if (near) {
        rank += 0.2;
      } else if (s < kBallThreshold) {
        continue; // weak detections only count near the tracked ball
      }
    }

    if (rank > bestRank) {
      bestRank = rank;
      bestScore = s;
      best = c;
      bcx = ux;
      bcy = uy;
      bw = w / lb.scale;
      bh = h / lb.scale;
    }
  }

  if (best < 0) return;

  res[16] = bestScore;
  res[17] = bcx;
  res[18] = bcy;
  res[19] = bw;
  res[20] = bh;
}

// ================================================================
// TAP DETECTION (per foot)
// ================================================================

class _FootTap {
  // true  = foot has been away from the ball since its last tap and may
  //         count again on its next approach.
  // false = foot already counted (or started on the ball) and must first
  //         move away.
  bool armed = false;
  double? lastDist; // previous frame's distance, to know it is approaching
  double minSince = double.infinity; // closest point since the last count
  DateTime lastCount = DateTime.fromMillisecondsSinceEpoch(0);

  void reset() {
    armed = false;
    lastDist = null;
    minSince = double.infinity;
  }
}

// ================================================================
// PAGE
// ================================================================

class ToeTapPage extends StatefulWidget {
  const ToeTapPage({super.key, required this.camera});

  final CameraDescription camera;

  @override
  State<ToeTapPage> createState() => _ToeTapPageState();
}

class _ToeTapPageState extends State<ToeTapPage> with WidgetsBindingObserver {
  // ---- Tap detection tuning ----
  // COCO has no toe keypoint, so a second "toe" point is the ankle pushed
  // this far along the knee->ankle direction (fraction of shin length).
  static const double footExtend = 0.30;
  // Contact zone around the ball centre:
  //   contactR = ballRadius * contactBallFactor + shin * contactShinFactor
  static const double contactBallFactor = 1.25;
  static const double contactShinFactor = 0.15;
  // CONTACT: counted the moment an approaching foot comes this close
  // to the ball centre (in contactR units).
  static const double touchFactor = 0.90;
  // RE-ARM after a count: the foot must move away, i.e. either leave the
  // zone (distance > rearmFactor * contactR) ...
  static const double rearmFactor = 1.10;
  // ... or back off by this much from its closest point since the count
  // (needed when the foot hovers over the ball in the image).
  static const double retreatFactor = 0.40;
  // Minimum time between two counts of the same foot.
  static const int footCooldownMs = 250;
  // Safety net against both feet counting the same moment.
  static const int cooldownMs = 150;
  // Ball is often hidden by the foot at contact, and is only detected every
  // kBallEvery-th frame, so remember its last position for a while.
  static const int ballHoldMs = 600 * kBallEvery;

  // ---- Worker isolate ----
  Isolate? _isolate;
  ReceivePort? _fromWorker;
  SendPort? _toWorker;
  bool _workerReady = false;
  bool _ballEnabled = true;
  bool _busy = false;

  // ---- Camera ----
  CameraController? _camera;
  bool _startingCamera = false;

  // ---- UI state ----
  bool _loading = true;
  bool _running = false;
  String _status = 'Loading models...';

  Size _upright = const Size(480, 640);

  double _personConfidence = 0.0;
  double _inferenceMs = 0.0;

  Offset? _leftAnkle;
  Offset? _rightAnkle;
  Offset? _leftKnee;
  Offset? _rightKnee;
  Offset? _leftFoot;
  Offset? _rightFoot;

  Offset? _ball; // smoothed centre (upright px); null when not tracked
  double _ballR = 0.0; // smoothed radius (upright px)
  DateTime? _ballSeen;

  // ---- Tap state ----
  int _tapCount = 0;
  final _FootTap _leftTap = _FootTap();
  final _FootTap _rightTap = _FootTap();
  bool _needPrime = true;
  DateTime? _lastTapTime;
  double _shin = 0.0;
  double _proximity = 0.0; // nearest foot distance / contactR (<1 = contact)

  // ------------------------------------------------------------
  // INIT
  // ------------------------------------------------------------

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      final poseData = await rootBundle.load(kPoseAsset);
      final poseBytes = poseData.buffer.asUint8List(
        poseData.offsetInBytes,
        poseData.lengthInBytes,
      );

      Uint8List? ballBytes;
      try {
        final bd = await rootBundle.load(kBallAsset);
        ballBytes = bd.buffer.asUint8List(bd.offsetInBytes, bd.lengthInBytes);
      } catch (_) {
        debugPrint('Ball model $kBallAsset not found; running pose only.');
      }

      final ready = Completer<void>();
      final port = ReceivePort();
      _fromWorker = port;

      port.listen((dynamic msg) {
        if (msg is SendPort) {
          _toWorker = msg;
          msg.send({
            'type': 'init',
            'pose': TransferableTypedData.fromList([poseBytes]),
            'ball': ballBytes == null
                ? null
                : TransferableTypedData.fromList([ballBytes]),
          });
          return;
        }

        if (msg is Map) {
          switch (msg['type']) {
            case 'ready':
              _workerReady = true;
              _ballEnabled = msg['ball'] == true;
              if (!ready.isCompleted) ready.complete();
              break;
            case 'error':
              if (!ready.isCompleted) {
                ready.completeError(msg['message'] ?? 'worker error');
              }
              break;
            case 'result':
              _onResult(msg['data'] as Float64List);
              break;
            case 'frame_error':
              _busy = false;
              break;
          }
        }
      });

      _isolate = await Isolate.spawn(_workerMain, port.sendPort);

      await ready.future;
      await _startCamera();

      if (!mounted) return;

      setState(() {
        _loading = false;
        _status = _ballEnabled
            ? 'Ready. Press Start.'
            : 'Football model missing ($kBallAsset)';
      });
    } catch (e, st) {
      debugPrint('Initialization error: $e\n$st');

      if (!mounted) return;

      setState(() {
        _loading = false;
        _status = 'Initialization failed: $e';
      });
    }
  }

  // ------------------------------------------------------------
  // CAMERA
  // ------------------------------------------------------------

  Future<void> _startCamera() async {
    if (_startingCamera || _camera != null) return;
    _startingCamera = true;

    try {
      final controller = CameraController(
        widget.camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await controller.initialize();

      try {
        await controller.setFocusMode(FocusMode.auto);
        await controller.setExposureMode(ExposureMode.auto);
      } catch (_) {}

      final ps = controller.value.previewSize;
      if (ps != null) {
        final rot = widget.camera.sensorOrientation;
        final swap = rot == 90 || rot == 270;
        _upright = swap ? Size(ps.height, ps.width) : Size(ps.width, ps.height);
      }

      await controller.startImageStream(_onCameraImage);

      _camera = controller;

      if (mounted) setState(() {});
    } finally {
      _startingCamera = false;
    }
  }

  Future<void> _stopCamera() async {
    final c = _camera;
    _camera = null;

    if (mounted) setState(() {});

    if (c == null) return;

    try {
      if (c.value.isStreamingImages) {
        await c.stopImageStream();
      }
    } catch (_) {}

    try {
      await c.dispose();
    } catch (_) {}
  }

  void _onCameraImage(CameraImage image) {
    // Drop frames while the worker is busy: always work on the freshest frame.
    if (_busy || !_workerReady || _toWorker == null) return;

    _busy = true;

    final p = image.planes;
    final b = _ball;

    _toWorker!.send({
      'type': 'frame',
      'w': image.width,
      'h': image.height,
      'y': TransferableTypedData.fromList([p[0].bytes]),
      'u': TransferableTypedData.fromList([p[1].bytes]),
      'v': TransferableTypedData.fromList([p[2].bytes]),
      'yStride': p[0].bytesPerRow,
      'uvStride': p[1].bytesPerRow,
      'uvPixel': p[1].bytesPerPixel ?? 1,
      'rotation': widget.camera.sensorOrientation,
      // ball tracking hint
      'hx': b?.dx ?? 0.0,
      'hy': b?.dy ?? 0.0,
      'hr': b != null ? _ballR : 0.0,
    });
  }

  // ------------------------------------------------------------
  // RESULT HANDLING
  // ------------------------------------------------------------

  Offset? _ema(Offset? previous, Offset? current) {
    if (current == null) return null;
    if (previous == null) return current;

    const a = 0.6;

    return Offset(
      previous.dx + (current.dx - previous.dx) * a,
      previous.dy + (current.dy - previous.dy) * a,
    );
  }

  // COCO has no toe point: extend the knee->ankle line past the ankle.
  Offset? _foot(Offset? ankle, Offset? knee) {
    if (ankle == null) return null;
    if (knee == null) return ankle;
    return ankle + (ankle - knee) * footExtend;
  }

  void _onResult(Float64List d) {
    _busy = false;

    if (!mounted) return;

    final now = DateTime.now();
    final person = d[0];
    final uw = d[1];
    final uh = d[2];

    Offset? kp(int i, double minConf) {
      if (person < kPersonThreshold) return null;
      if (d[6 + i * 3] < minConf) return null;
      return Offset(d[4 + i * 3], d[5 + i * 3]);
    }

    final la = _ema(_leftAnkle, kp(0, kAnkleThreshold));
    final ra = _ema(_rightAnkle, kp(1, kAnkleThreshold));
    final lk = _ema(_leftKnee, kp(2, kKneeThreshold));
    final rk = _ema(_rightKnee, kp(3, kKneeThreshold));
    final lf = _foot(la, lk);
    final rf = _foot(ra, rk);

    _updateShin(la, lk, ra, rk);

    // Ball
    if (d[16] >= kBallLow) {
      _ball = _ema(_ball, Offset(d[17], d[18]));
      final r = (d[19] + d[20]) / 4.0;
      _ballR = _ballR == 0 ? r : _ballR * 0.7 + r * 0.3;
      _ballSeen = now;
    }

    final seen = _ballSeen;
    final ballFresh =
        seen != null && now.difference(seen).inMilliseconds < ballHoldMs;
    if (!ballFresh) _ball = null;
    final ball = _ball;

    if (_running) {
      _updateTapCounter(now, la, lf, ra, rf, ball);
    }

    setState(() {
      if (uw > 0 && uh > 0) _upright = Size(uw, uh);

      _personConfidence = person;
      _inferenceMs = d[3];

      _leftAnkle = la;
      _rightAnkle = ra;
      _leftKnee = lk;
      _rightKnee = rk;
      _leftFoot = lf;
      _rightFoot = rf;

      if (person < kPersonThreshold) {
        _status = 'Looking for person...';
      } else if (lf == null || rf == null) {
        _status = 'Show both feet';
      } else if (_ballEnabled && ball == null) {
        _status = 'Show the football';
      } else {
        _status = _running ? 'Counting...' : 'Ready. Press Start.';
      }
    });
  }

  void _updateShin(Offset? la, Offset? lk, Offset? ra, Offset? rk) {
    final shins = <double>[];
    if (la != null && lk != null) shins.add((la - lk).distance);
    if (ra != null && rk != null) shins.add((ra - rk).distance);

    if (shins.isNotEmpty) {
      final s = shins.reduce((a, b) => a + b) / shins.length;
      _shin = _shin == 0 ? s : _shin * 0.9 + s * 0.1;
    }
  }

  // ------------------------------------------------------------
  // TOE TAP DETECTION (ball contact)
  // ------------------------------------------------------------
  //
  // Per foot, distance = min(distance of ankle, distance of toe point)
  // to the ball centre. Each foot runs this cycle independently:
  //
  //   ARMED  foot approaches ball (distance shrinking)
  //            |  distance < touchFactor * contactR
  //            v
  //   CONTACT detected  ->  COUNT +1  ->  DISARMED
  //            |
  //   DISARMED foot sits on / hovers near the ball: NOTHING is counted,
  //            no matter how many frames pass
  //            |  foot moves away (leaves zone, or backs off from its
  //            |  closest point by retreatFactor)
  //            v
  //   ARMED  again, waiting for the next approach -> next contact -> +1
  //
  // So one approach = one tap, and the counter only moves on genuine
  // approach-and-touch events, never because a foot is merely close.
  // Left and right feet are independent, so alternating taps count too.
  // All sizes scale with ball radius and shin length, so it works at
  // any distance from the camera.
  // ------------------------------------------------------------

  double _footDistance(Offset? ankle, Offset? toe, Offset ball) {
    double best = double.infinity;
    if (ankle != null) best = math.min(best, (ankle - ball).distance);
    if (toe != null) best = math.min(best, (toe - ball).distance);
    return best;
  }

  /// Returns true on the frame a NEW contact is detected for this foot.
  bool _stepFoot(_FootTap f, double dist, double contactR, DateTime now) {
    final prev = f.lastDist;
    f.lastDist = dist;

    if (!f.armed) {
      // Waiting for the foot to move away from the last contact.
      f.minSince = math.min(f.minSince, dist);

      final leftZone = dist > contactR * rearmFactor;
      final backedOff = f.minSince.isFinite &&
          dist > f.minSince + contactR * retreatFactor;

      if (leftZone || backedOff) {
        f.armed = true;
        f.minSince = double.infinity;
      }
      return false;
    }

    // Armed: contact = approaching AND close enough.
    final approaching = prev != null && dist < prev;
    final touching = dist < contactR * touchFactor;
    final cooled = now.difference(f.lastCount).inMilliseconds > footCooldownMs;

    if (approaching && touching && cooled) {
      f.armed = false;
      f.minSince = dist;
      f.lastCount = now;
      return true; // COUNT at the moment of contact
    }

    return false;
  }

  void _updateTapCounter(
    DateTime now,
    Offset? leftAnkle,
    Offset? leftToe,
    Offset? rightAnkle,
    Offset? rightToe,
    Offset? ball,
  ) {
    if (ball == null) {
      _leftTap.reset();
      _rightTap.reset();
      _needPrime = true;
      return;
    }

    final shin = _shin > 20 ? _shin : _upright.height * 0.12;
    final contactR = _ballR * contactBallFactor + shin * contactShinFactor;

    final dl = _footDistance(leftAnkle, leftToe, ball);
    final dr = _footDistance(rightAnkle, rightToe, ball);

    // After Start/Reset/ball re-acquired both feet begin DISARMED: a foot
    // already on the ball must move away before it can count. A foot that
    // is far away arms itself on the very first frame.
    if (_needPrime) {
      _leftTap.reset();
      _rightTap.reset();
      _needPrime = false;
    }

    bool tap = false;
    if (dl.isFinite) tap = _stepFoot(_leftTap, dl, contactR, now) || tap;
    if (dr.isFinite) tap = _stepFoot(_rightTap, dr, contactR, now) || tap;

    if (tap) {
      final last = _lastTapTime;
      if (last == null || now.difference(last).inMilliseconds > cooldownMs) {
        _tapCount++;
        _lastTapTime = now;
      }
    }

    final nearest = math.min(dl, dr);
    _proximity = nearest.isFinite ? nearest / contactR : 0.0;
  }

  // ------------------------------------------------------------
  // START / RESET
  // ------------------------------------------------------------

  void _toggleRunning() {
    setState(() {
      _running = !_running;
      _leftTap.reset();
      _rightTap.reset();
      _needPrime = true;
      _status = _running ? 'Counting...' : 'Paused';
    });
  }

  void _reset() {
    setState(() {
      _tapCount = 0;
      _leftTap.reset();
      _rightTap.reset();
      _needPrime = true;
      _lastTapTime = null;
      _status = _running ? 'Counting...' : 'Ready. Press Start.';
    });
  }

  // ------------------------------------------------------------
  // LIFECYCLE
  // ------------------------------------------------------------

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (!_workerReady) return;

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _stopCamera();
    } else if (state == AppLifecycleState.resumed) {
      _startCamera().catchError((e) {
        debugPrint('Camera resume error: $e');
      });
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);

    final c = _camera;
    _camera = null;

    if (c != null) {
      () async {
        try {
          if (c.value.isStreamingImages) await c.stopImageStream();
        } catch (_) {}
        try {
          await c.dispose();
        } catch (_) {}
      }();
    }

    _isolate?.kill(priority: Isolate.immediate);
    _fromWorker?.close();

    super.dispose();
  }

  // ------------------------------------------------------------
  // UI
  // ------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final camera = _camera;

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (camera != null && camera.value.isInitialized)
              ClipRect(
                child: FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: _upright.width,
                    height: _upright.height,
                    child: CameraPreview(camera),
                  ),
                ),
              )
            else
              const Center(child: CircularProgressIndicator()),

            IgnorePointer(
              child: CustomPaint(
                painter: _PosePainter(
                  leftKnee: _leftKnee,
                  rightKnee: _rightKnee,
                  leftAnkle: _leftAnkle,
                  rightAnkle: _rightAnkle,
                  leftFoot: _leftFoot,
                  rightFoot: _rightFoot,
                  ball: _ball,
                  ballRadius: _ballR,
                  imageSize: _upright,
                ),
              ),
            ),

            Positioned(
              top: 16,
              left: 16,
              right: 16,
              child: _infoPanel(),
            ),

            Positioned(
              left: 16,
              right: 16,
              bottom: 20,
              child: Row(
                children: [
                  Expanded(
                    child: FilledButton(
                      onPressed: _loading ? null : _toggleRunning,
                      child: Text(_running ? 'Stop' : 'Start'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton(
                      onPressed: _reset,
                      child: const Text('Reset'),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _infoPanel() {
    final fps = _inferenceMs > 0 ? 1000.0 / _inferenceMs : 0.0;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.72),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Text(
                'Toe Taps',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              const Spacer(),
              Text(
                '$_tapCount',
                style: const TextStyle(
                  fontSize: 30,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(_status),
          const SizedBox(height: 4),
          Text(
            'Person ${_personConfidence.toStringAsFixed(2)}'
            '  •  ${_inferenceMs.toStringAsFixed(0)} ms'
            ' (${fps.toStringAsFixed(0)} fps)'
            '  •  ball ${_ball != null ? "yes" : "no"}'
            '  •  foot/ball ${_proximity.toStringAsFixed(2)} (<1 = touch)',
            style: const TextStyle(fontSize: 12),
          ),
        ],
      ),
    );
  }
}

// ================================================================
// OVERLAY
// ================================================================

class _PosePainter extends CustomPainter {
  const _PosePainter({
    required this.leftKnee,
    required this.rightKnee,
    required this.leftAnkle,
    required this.rightAnkle,
    required this.leftFoot,
    required this.rightFoot,
    required this.ball,
    required this.ballRadius,
    required this.imageSize,
  });

  final Offset? leftKnee;
  final Offset? rightKnee;
  final Offset? leftAnkle;
  final Offset? rightAnkle;
  final Offset? leftFoot;
  final Offset? rightFoot;
  final Offset? ball;
  final double ballRadius;

  // UPRIGHT image size (same size the preview is cover-fitted from).
  final Size imageSize;

  double _scaleFor(Size screen) => math.max(
        screen.width / imageSize.width,
        screen.height / imageSize.height,
      );

  Offset? _toScreen(Offset? p, Size screen) {
    if (p == null || imageSize.width <= 0 || imageSize.height <= 0) {
      return null;
    }

    final scale = _scaleFor(screen);
    final dx = (screen.width - imageSize.width * scale) / 2.0;
    final dy = (screen.height - imageSize.height * scale) / 2.0;

    return Offset(p.dx * scale + dx, p.dy * scale + dy);
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.width <= 0 || imageSize.height <= 0) return;

    final line = Paint()
      ..color = Colors.white.withValues(alpha: 0.6)
      ..strokeWidth = 3;

    void segment(Offset? a, Offset? b) {
      final pa = _toScreen(a, size);
      final pb = _toScreen(b, size);
      if (pa != null && pb != null) canvas.drawLine(pa, pb, line);
    }

    void dot(Offset? p, String label, Color color, double radius) {
      final s = _toScreen(p, size);
      if (s == null) return;

      canvas.drawCircle(
        s,
        radius + 4,
        Paint()..color = Colors.black.withValues(alpha: 0.75),
      );
      canvas.drawCircle(s, radius, Paint()..color = color);

      if (label.isEmpty) return;

      final tp = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.black,
            fontSize: 10,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      tp.paint(canvas, s - Offset(tp.width / 2, tp.height / 2));
    }

    // Ball
    final b = _toScreen(ball, size);
    if (b != null && ballRadius > 0) {
      canvas.drawCircle(
        b,
        ballRadius * _scaleFor(size),
        Paint()
          ..color = Colors.orangeAccent
          ..style = PaintingStyle.stroke
          ..strokeWidth = 4,
      );
      dot(ball, '', Colors.orangeAccent, 4);
    }

    // Legs: knee -> ankle -> toe point
    segment(leftKnee, leftAnkle);
    segment(rightKnee, rightAnkle);
    segment(leftAnkle, leftFoot);
    segment(rightAnkle, rightFoot);

    dot(leftKnee, '', Colors.white, 4);
    dot(rightKnee, '', Colors.white, 4);
    dot(leftAnkle, '', Colors.white, 4);
    dot(rightAnkle, '', Colors.white, 4);
    dot(leftFoot, 'L', Colors.greenAccent, 10);
    dot(rightFoot, 'R', Colors.redAccent, 10);
  }

  @override
  bool shouldRepaint(covariant _PosePainter old) => true;
}