import 'dart:math' as math;
import '../config/app_config.dart';

/// =============================================================================
/// Project Vani: Landmark Coordinate Extractor & Preprocessor
/// =============================================================================
class FrameProcessor {
  /// Matrix Dimension Constants:
  /// - Pose:      33 * 4 = 132
  /// - Face:      468 * 3 = 1404
  /// - Left Hand: 21 * 3 = 63
  /// - Right Hand:21 * 3 = 63
  /// Total: 1662 float values
  static const int poseDim = 132;
  static const int faceDim = 1404;
  static const int handDim = 63;
  static const int totalDim = AppConfig.featureDimension;

  /// Concatenates individual landmark subsets into a unified 1662 vector with zero-padding
  static List<double> formatLandmarkVector({
    List<double>? poseLandmarks,
    List<double>? faceLandmarks,
    List<double>? leftHandLandmarks,
    List<double>? rightHandLandmarks,
  }) {
    final List<double> fullVector = [];

    // 1. Pose (132 features)
    if (poseLandmarks != null && poseLandmarks.isNotEmpty) {
      if (poseLandmarks.length >= poseDim) {
        fullVector.addAll(poseLandmarks.sublist(0, poseDim));
      } else {
        fullVector.addAll(poseLandmarks);
        fullVector.addAll(List<double>.filled(poseDim - poseLandmarks.length, 0.0));
      }
    } else {
      fullVector.addAll(List<double>.filled(poseDim, 0.0));
    }

    // 2. Face (1404 features)
    if (faceLandmarks != null && faceLandmarks.isNotEmpty) {
      if (faceLandmarks.length >= faceDim) {
        fullVector.addAll(faceLandmarks.sublist(0, faceDim));
      } else {
        fullVector.addAll(faceLandmarks);
        fullVector.addAll(List<double>.filled(faceDim - faceLandmarks.length, 0.0));
      }
    } else {
      fullVector.addAll(List<double>.filled(faceDim, 0.0));
    }

    // 3. Left Hand (63 features)
    if (leftHandLandmarks != null && leftHandLandmarks.isNotEmpty) {
      if (leftHandLandmarks.length >= handDim) {
        fullVector.addAll(leftHandLandmarks.sublist(0, handDim));
      } else {
        fullVector.addAll(leftHandLandmarks);
        fullVector.addAll(List<double>.filled(handDim - leftHandLandmarks.length, 0.0));
      }
    } else {
      fullVector.addAll(List<double>.filled(handDim, 0.0));
    }

    // 4. Right Hand (63 features)
    if (rightHandLandmarks != null && rightHandLandmarks.isNotEmpty) {
      if (rightHandLandmarks.length >= handDim) {
        fullVector.addAll(rightHandLandmarks.sublist(0, handDim));
      } else {
        fullVector.addAll(rightHandLandmarks);
        fullVector.addAll(List<double>.filled(handDim - rightHandLandmarks.length, 0.0));
      }
    } else {
      fullVector.addAll(List<double>.filled(handDim, 0.0));
    }

    assert(fullVector.length == totalDim, 'Vector length must equal $totalDim');
    return fullVector;
  }

  /// Generates a synthetic landmark frame vector for testing and emulation
  static List<double> generateMockFrame(double phase) {
    final frame = List<double>.filled(totalDim, 0.0);
    // Simulate active hand motion
    for (int i = 0; i < handDim; i++) {
      frame[poseDim + faceDim + i] = (math.sin(phase + i * 0.1) + 1.0) / 2.0;
    }
    return frame;
  }
}
