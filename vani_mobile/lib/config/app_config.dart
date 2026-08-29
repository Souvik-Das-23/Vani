/// =============================================================================
/// Project Vani: Mobile Application Configuration
/// =============================================================================
class AppConfig {
  /// Default backend URL for Android Emulator (10.0.2.2) and local testing
  static String baseUrl = "http://10.0.2.2:8000/api";

  /// Number of sequential frames expected by the LSTM model
  static const int sequenceLength = 30;

  /// Total flattened landmark keypoint coordinates per frame (132 + 1404 + 63 + 63)
  static const int featureDimension = 1662;

  /// Cooldown between consecutive API inference calls to prevent network congestion
  static const int debounceIntervalMs = 1200;

  /// Minimum confidence threshold to accept gesture into sentence history
  static const double confidenceThreshold = 0.65;

  /// Target stream FPS processing rate
  static const int targetFps = 30;

  /// Endpoint URLs
  static String get translateUrl => "$baseUrl/translate/";
  static String get healthUrl => "$baseUrl/health/";
  static String get actionsUrl => "$baseUrl/actions/";
}
