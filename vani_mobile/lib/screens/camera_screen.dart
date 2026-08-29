import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';

import '../config/app_config.dart';
import '../models/translation_result.dart';
import '../services/api_service.dart';
import '../services/landmark_buffer.dart';
import '../services/frame_processor.dart';
import 'widgets/translation_overlay.dart';
import 'widgets/sentence_bar.dart';
import 'widgets/settings_dialog.dart';

/// =============================================================================
/// Project Vani: Fullscreen Camera Translation Screen
/// =============================================================================
class CameraScreen extends StatefulWidget {
  const CameraScreen({Key? key}) : super(key: key);

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with WidgetsBindingObserver {
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  int _selectedCameraIndex = 0;

  bool _isCameraInitialized = false;
  bool _isStreaming = false;
  bool _isProcessingApi = false;
  bool _isServerOnline = false;

  final ApiService _apiService = ApiService();
  final LandmarkBuffer _buffer = LandmarkBuffer();

  TranslationResult _currentResult = TranslationResult.empty();
  final List<String> _sentence = [];

  DateTime _lastApiCallTime = DateTime.now();
  DateTime _lastSentenceUpdateTime = DateTime.now();
  Timer? _healthCheckTimer;
  double _mockPhase = 0.0; // Simulation phase for stream ticks

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initApp();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _healthCheckTimer?.cancel();
    _cameraController?.dispose();
    _apiService.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }
    if (state == AppLifecycleState.inactive) {
      controller.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera(_selectedCameraIndex);
    }
  }

  Future<void> _initApp() async {
    await _checkBackendHealth();
    // Recurring health check every 10 seconds
    _healthCheckTimer = Timer.periodic(const Duration(seconds: 10), (_) {
      _checkBackendHealth();
    });
    await _requestPermissionAndInitCamera();
  }

  Future<void> _checkBackendHealth() async {
    final online = await _apiService.checkHealth();
    if (mounted) {
      setState(() {
        _isServerOnline = online;
      });
    }
  }

  Future<void> _requestPermissionAndInitCamera() async {
    final status = await Permission.camera.request();
    if (status.isGranted) {
      try {
        _cameras = await availableCameras();
        if (_cameras.isNotEmpty) {
          // Select front camera if available
          final frontCameraIdx = _cameras.indexWhere(
            (c) => c.lensDirection == CameraLensDirection.front,
          );
          _selectedCameraIndex = frontCameraIdx != -1 ? frontCameraIdx : 0;
          await _initCamera(_selectedCameraIndex);
        }
      } catch (e) {
        debugPrint("Camera initialization error: $e");
      }
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Camera permission is required for sign language recognition."),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
    }
  }

  Future<void> _initCamera(int cameraIndex) async {
    if (_cameras.isEmpty) return;

    final controller = CameraController(
      _cameras[cameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    _cameraController = controller;

    try {
      await controller.initialize();
      if (!mounted) return;

      setState(() {
        _isCameraInitialized = true;
      });

      _startImageProcessingStream();
    } catch (e) {
      debugPrint("Failed to initialize camera controller: $e");
    }
  }

  void _startImageProcessingStream() {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) return;

    _isStreaming = true;
    controller.startImageStream((CameraImage image) {
      _processCameraFrame(image);
    });
  }

  /// Extracts coordinates, feeds 30-frame buffer, and triggers debounced API calls
  void _processCameraFrame(CameraImage image) {
    // 1. Synthesize/Extract 1662 keypoints from frame
    _mockPhase += 0.15;
    final frameVector = FrameProcessor.generateMockFrame(_mockPhase);

    // 2. Buffer into 30-frame sliding window
    _buffer.addFrame(frameVector);

    // Update UI buffer progress
    if (mounted && !_buffer.isReady) {
      setState(() {});
    }

    // 3. Debounce & Trigger Inference Request
    final now = DateTime.now();
    final elapsedMs = now.difference(_lastApiCallTime).inMilliseconds;

    if (_buffer.isReady &&
        elapsedMs >= AppConfig.debounceIntervalMs &&
        !_isProcessingApi) {
      _lastApiCallTime = now;
      _sendInferenceRequest();
    }
  }

  Future<void> _sendInferenceRequest() async {
    if (_isProcessingApi) return;
    _isProcessingApi = true;

    final sequence = _buffer.getSequence();
    if (sequence.isEmpty) {
      _isProcessingApi = false;
      return;
    }

    try {
      final result = await _apiService.translateSequence(sequence);

      if (mounted) {
        setState(() {
          _currentResult = result;
          _isProcessingApi = false;
        });

        // 4. Update accumulated sentence with debounce
        final now = DateTime.now();
        if (result.isConfident &&
            now.difference(_lastSentenceUpdateTime).inMilliseconds > 1500) {
          if (_sentence.isEmpty || _sentence.last != result.action) {
            setState(() {
              _sentence.add(result.action);
              _lastSentenceUpdateTime = now;
            });
          }
        }
      }
    } catch (e) {
      debugPrint("Inference failed: $e");
      if (mounted) {
        setState(() {
          _isProcessingApi = false;
        });
      }
    }
  }

  void _switchCamera() async {
    if (_cameras.length < 2) return;
    final nextIdx = (_selectedCameraIndex + 1) % _cameras.length;
    _selectedCameraIndex = nextIdx;

    setState(() {
      _isCameraInitialized = false;
    });

    await _cameraController?.dispose();
    await _initCamera(_selectedCameraIndex);
  }

  void _showSettingsDialog() {
    showDialog(
      context: context,
      builder: (context) => BackendSettingsDialog(
        onConfigSaved: () {
          _checkBackendHealth();
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. Fullscreen Camera Preview
          if (_isCameraInitialized && _cameraController != null)
            Center(
              child: CameraPreview(_cameraController!),
            )
          else
            const Center(
              child: CircularProgressIndicator(
                color: Color(0xFF00E5FF),
              ),
            ),

          // 2. Top Header Bar (Brand, Status & Actions)
          Positioned(
            top: 50,
            left: 20,
            right: 20,
            child: _buildTopHeader(),
          ),

          // 3. Center Target Viewfinder Overlay
          Center(
            child: Container(
              width: MediaQuery.of(context).size.width * 0.75,
              height: MediaQuery.of(context).size.height * 0.40,
              decoration: BoxDecoration(
                border: Border.all(
                  color: Colors.white.withOpacity(0.15),
                  width: 1.5,
                ),
                borderRadius: BorderRadius.circular(24),
              ),
            ),
          ),

          // 4. Bottom Translation Overlays
          Positioned(
            bottom: 30,
            left: 20,
            right: 20,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Live Prediction Card
                TranslationOverlay(
                  result: _currentResult,
                  bufferProgress: _buffer.progress,
                  isBuffering: !_buffer.isReady,
                ),
                const SizedBox(height: 12),

                // Sentence History Banner
                SentenceBar(
                  sentence: _sentence,
                  onClear: () {
                    setState(() {
                      _sentence.clear();
                    });
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTopHeader() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        // App Branding
        Row(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFF1E1E2E).withOpacity(0.85),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.white12),
              ),
              child: Row(
                children: [
                  const Text(
                    "VANI",
                    style: TextStyle(
                      color: Color(0xFF00E5FF),
                      fontWeight: FontWeight.w900,
                      fontSize: 16,
                      letterSpacing: 1.5,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _isServerOnline
                          ? const Color(0xFF00E676)
                          : const Color(0xFFFF5252),
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    _isServerOnline ? "ONLINE" : "OFFLINE",
                    style: TextStyle(
                      color: _isServerOnline
                          ? const Color(0xFF00E676)
                          : const Color(0xFFFF5252),
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),

        // Action Buttons
        Row(
          children: [
            _buildIconButton(
              icon: Icons.flip_camera_ios_rounded,
              tooltip: "Switch Camera",
              onTap: _switchCamera,
            ),
            const SizedBox(width: 10),
            _buildIconButton(
              icon: Icons.tune_rounded,
              tooltip: "Backend Settings",
              onTap: _showSettingsDialog,
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildIconButton({
    required IconData icon,
    required String tooltip,
    required VoidCallback onTap,
  }) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Material(
        color: const Color(0xFF1E1E2E).withOpacity(0.85),
        child: InkWell(
          onTap: onTap,
          child: Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.white12),
            ),
            child: Icon(icon, color: Colors.white, size: 20),
          ),
        ),
      ),
    );
  }
}
