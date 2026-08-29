/// =============================================================================
/// Project Vani: Translation Response Data Model
/// =============================================================================
class TranslationResult {
  final String status;
  final String action;
  final double confidence;
  final Map<String, double> probabilities;
  final int sequenceLength;
  final double latencyMs;

  TranslationResult({
    required this.status,
    required this.action,
    required this.confidence,
    required this.probabilities,
    required this.sequenceLength,
    required this.latencyMs,
  });

  factory TranslationResult.fromJson(Map<String, dynamic> json) {
    final rawProb = json['probabilities'] as Map<String, dynamic>? ?? {};
    final Map<String, double> probMap = {};
    rawProb.forEach((key, value) {
      probMap[key] = (value as num).toDouble();
    });

    return TranslationResult(
      status: json['status'] ?? 'unknown',
      action: json['action'] ?? '...',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      probabilities: probMap,
      sequenceLength: json['sequence_length'] ?? 30,
      latencyMs: (json['latency_ms'] as num?)?.toDouble() ?? 0.0,
    );
  }

  factory TranslationResult.empty() {
    return TranslationResult(
      status: 'idle',
      action: '...',
      confidence: 0.0,
      probabilities: {},
      sequenceLength: 0,
      latencyMs: 0.0,
    );
  }

  bool get isConfident => confidence >= 0.65 && action != '...';
}
