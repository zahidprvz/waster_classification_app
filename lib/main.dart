import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:io';
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EcoSort - Waste Classification',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSwatch(
          primarySwatch: Colors.teal,
          accentColor: Colors.tealAccent,
        ),
        textTheme: const TextTheme(
          bodyMedium: TextStyle(color: Colors.white),
        ),
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const ImageClassifier(),
    );
  }
}

class ImageClassifier extends StatefulWidget {
  const ImageClassifier({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _ImageClassifierState createState() => _ImageClassifierState();
}

class _ImageClassifierState extends State<ImageClassifier> {
  Interpreter? _interpreter;
  File? _image;
  String _result = '';
  final ImagePicker _picker = ImagePicker();
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter =
          await Interpreter.fromAsset('waste_classifier_model.tflite');
      print('Model loaded successfully');
    } catch (e) {
      print('Failed to load the model: $e');
    }
  }

  Future<void> _classifyImage(File image) async {
    setState(() {
      _isProcessing = true;
    });

    if (_interpreter == null) {
      print('Interpreter not initialized');
      return;
    }

    var imageBytes = await image.readAsBytes();
    img.Image? imageAsImage = img.decodeJpg(imageBytes);

    var inputImage = img.copyResize(imageAsImage!, width: 224, height: 224);

    var inputAsList = List<double>.empty(growable: true);

    for (int y = 0; y < inputImage.height; y++) {
      for (int x = 0; x < inputImage.width; x++) {
        int pixel = inputImage.getPixel(x, y);
        inputAsList.add(((pixel >> 16) & 0xFF) / 255.0); // Red channel
        inputAsList.add(((pixel >> 8) & 0xFF) / 255.0); // Green channel
        inputAsList.add((pixel & 0xFF) / 255.0); // Blue channel
      }
    }

    var inputTensor =
        Float32List.fromList(inputAsList).reshape([1, 224, 224, 3]);

    var output = List.filled(1 * 2, 0.0).reshape([1, 2]);

    // Run inference in the background
    await Future.delayed(Duration(milliseconds: 2000), () async {
      // Show the loading indicator
      setState(() {
        _isProcessing = true;
      });

      // Run inference
      _interpreter!.run(inputTensor, output);

      // Hide the loading indicator
      setState(() {
        _isProcessing = false;
        _result = output[0][0] > output[0][1] ? 'Organic' : 'Recyclable';
      });
    });
  }

  Future<void> _pickImage() async {
    final pickedImage = await _picker.pickImage(source: ImageSource.camera);
    if (pickedImage != null) {
      setState(() {
        _image = File(pickedImage.path);
      });
      await _classifyImage(_image!);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('EcoSort - Classify Your Waste'),
        centerTitle: true,
      ),
      body: Container(
        padding: const EdgeInsets.symmetric(vertical: 16.0),
        alignment: Alignment.center,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topRight,
            end: Alignment.bottomLeft,
            stops: const [0.1, 0.5, 0.7, 0.9],
            colors: [
              Colors.teal[800]!,
              Colors.teal[700]!,
              Colors.teal[600]!,
              Colors.teal[400]!,
            ],
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            if (_image != null)
              Container(
                margin: const EdgeInsets.all(8.0),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8.0),
                  child: Image.file(_image!),
                ),
              ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: _isProcessing
                  ? const CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    )
                  : Text(
                      _result,
                      style: Theme.of(context)
                          .textTheme
                          .headlineMedium
                          ?.copyWith(color: Colors.white70),
                    ),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _pickImage,
        tooltip: 'Pick Image',
        icon: const Icon(Icons.camera),
        label: const Text('Capture Image'),
        backgroundColor: Colors.tealAccent[400],
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}
