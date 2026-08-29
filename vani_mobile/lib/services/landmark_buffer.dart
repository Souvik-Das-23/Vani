import 'dart:collection';
import '../config/app_config.dart';

/// =============================================================================
/// Project Vani: Temporal Sliding Window Sequence Buffer
/// =============================================================================
/// Maintains a circular FIFO buffer of exactly 30 frames of 1662 keypoints.
class LandmarkBuffer {
  final int capacity;
  final int featureDimension;
  final Queue<List<double>> _queue;

  LandmarkBuffer({
    this.capacity = AppConfig.sequenceLength,
    this.featureDimension = AppConfig.featureDimension,
  }) : _queue = Queue<List<double>>();

  /// Adds a new frame vector to the sliding window
  void addFrame(List<double> frame) {
    // Validate or pad frame to ensure invariant 1662 dimensions
    List<double> formattedFrame;
    if (frame.length == featureDimension) {
      formattedFrame = frame;
    } else if (frame.length < featureDimension) {
      formattedFrame = List<double>.from(frame)
        ..addAll(List<double>.filled(featureDimension - frame.length, 0.0));
    } else {
      formattedFrame = frame.sublist(0, featureDimension);
    }

    if (_queue.length >= capacity) {
      _queue.removeFirst();
    }
    _queue.addLast(formattedFrame);
  }

  /// Returns true when the buffer has accumulated a full 30-frame sequence
  bool get isReady => _queue.length == capacity;

  /// Returns the current buffer filling percentage (0.0 to 1.0)
  double get progress => _queue.length / capacity;

  /// Returns current frame count in the buffer
  int get length => _queue.length;

  /// Retrieves the sequence matrix as a List of 30 frames
  List<List<double>> getSequence({bool padIfIncomplete = false}) {
    if (isReady) {
      return List<List<double>>.from(_queue);
    }

    if (!padIfIncomplete) {
      return [];
    }

    final missing = capacity - _queue.length;
    final List<List<double>> padded = [];
    for (int i = 0; i < missing; i++) {
      padded.add(List<double>.filled(featureDimension, 0.0));
    }
    padded.addAll(_queue);
    return padded;
  }

  /// Clears all stored frames
  void clear() {
    _queue.clear();
  }
}
