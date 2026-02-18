import 'package:flutter_embedder/flutter_embedder.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path/path.dart' as path;
import 'package:image/image.dart' as img;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'dart:async';
import 'dart:math';
import 'dart:convert';

var initProc = false;
final initialized = Completer<bool>();

class MlService {
  late final dynamic config;
  late final HfTokenizer tokenizer;
  late final OnnxRuntime ort;
  late final OrtSession txtSess;
  late final OrtSession visSess;
  late final String txtX;
  late final String txtY;
  late final String visX;
  late final String visY;

  MlService() {
    if(initProc) {
      debugPrint('MLSERVICE ALREADY INIT');
      return;
    }
    initProc = true;
    init();
  }

  Future<void> init() async {
    await loadConfig();
    await initFlutterEmbedder();

    tokenizer = HfTokenizer.fromFile(path.join(config['mlPath'], 'tokenizer.json'));
    ort = OnnxRuntime();

    txtSess = await ort.createSession(path.join(config['mlPath'], 'textual.onnx'));
    txtX = txtSess.inputNames[0];
    txtY = txtSess.outputNames[0];

    visSess = await ort.createSession(path.join(config['mlPath'], 'visual.onnx'));
    visX = visSess.inputNames[0];
    visY = visSess.outputNames[0];

    initialized.complete(true);
  }

  Future<void> loadConfig() async {
    OrtDataType dtypeFromString(String s) {
      return switch(s) {
        'int32' => OrtDataType.int32,
        'int64' => OrtDataType.int64,
        'float32' => OrtDataType.float32,
        _ => OrtDataType.float32,
      };
    }

    dynamic similarityFromString(String s) {
      return switch(s) {
        'cosineSimilarity' => cosineSimilarity,
        'cosineSimilarityNormalized' => cosineSimilarityNormalized,
        _ => cosineSimilarity
      };
    }

    config = jsonDecode(await rootBundle.loadString('assets/mlconfig.json'));
    config['txtInputDType'] = dtypeFromString(config['txtInputDType']);
    config['visInputDType'] = dtypeFromString(config['visInputDType']);
    config['similarity'] = similarityFromString(config['similarity']);

    debugPrint('CONFIG ${config['visInputDType']}');
  }

  Future<Uint8List> txtInference(String query) async {
    await initialized.future;

    var h = [...tokenizer.encode(query).ids];
    h.addAll(List<int>.filled((config['contextLen'] - h.length).toInt(), 0));
    var j = await OrtValue.fromList(h, [1, config['contextLen']]);

    var k = await j.to(config['txtInputDType']);

    var y = await txtSess.run({txtX: k});
    var l = await y[txtY]!.to(OrtDataType.float32);
    var o = (await l.asList())[0];

    j.dispose();
    k.dispose();
    l.dispose();
    y.forEach((_, v) => v.dispose());

    return o.buffer.asUint8List();
  }

  Future<Uint8List> imgInference(String p) async {
    await initialized.future;

    final cmd = img.Command()
      ..decodeImageFile(p)
      ..copyResize(width: config['imgSize'], height: config['imgSize'], maintainAspect: true, interpolation: img.Interpolation.cubic)
      ..convert(numChannels: 3);
    await cmd.executeThread();

    var h = (await cmd.getImage())!
      .getBytes(order: img.ChannelOrder.rgb)
      .map((i) => i / 255.0)
      .toList();
    var j = await OrtValue.fromList(h, [1, (3 * config['imgSize'] * config['imgSize']).toInt()]);

    var k = await j.to(config['visInputDType']);

    var y = await visSess.run({visX: k});
    var l = await y[visY]!.to(OrtDataType.float32);
    var o = (await l.asList())[0];

    j.dispose();
    k.dispose();
    l.dispose();
    y.forEach((_, v) => v.dispose());

    return o.buffer.asUint8List();
  }

  (Float32List, Float32List) convertBlobToList(Uint8List tRaw, Uint8List vRaw) {
    ByteData b;

    final t = Float32List(config['embeddingLen']);
    final v = Float32List(config['embeddingLen']);

    b = ByteData.view(tRaw.buffer);
    for(int i = 0; i < config['embeddingLen']; i++) {
      t[i] = b.getFloat32(tRaw.offsetInBytes + 4*i, Endian.little);
    }
    b = ByteData.view(vRaw.buffer);
    for(int i = 0; i < config['embeddingLen']; i++) {
      v[i] = b.getFloat32(vRaw.offsetInBytes + 4*i, Endian.little);
    }

    return (t, v);
  }

  double cosineSimilarityNormalized(List t, List v) {
    double s = 0;

    for(var i = 0; i < config['embeddingLen']; i++) {
      s += t[i] * v[i];
    }
    return s;
  }

  double cosineSimilarity(List t, List v) {
    double dotP = 0.0;
    double mag1 = 0.0;
    double mag2 = 0.0;

    for(int i = 0; i < config['embeddingLen']; i++) {
      dotP += t[i] * v[i];
      mag1 += t[i] * t[i];
      mag2 += v[i] * v[i];
    }

    mag1 = sqrt(mag1);
    mag2 = sqrt(mag2);

    if(mag1 == 0.0 || mag2 == 0.0) return 0.0;
    return dotP / (mag1 * mag2);
  }

  double calcImgSimilarity(Uint8List tBlob, Uint8List vBlob) {
    var (t, v) = convertBlobToList(tBlob, vBlob);
    return config['similarity'](t, v);
  }
}