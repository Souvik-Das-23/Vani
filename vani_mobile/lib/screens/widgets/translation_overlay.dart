import 'dart:ui';
import 'package:flutter/material.dart';
import '../../models/translation_result.dart';

/// =============================================================================
/// Project Vani: Real-Time Translation HUD Overlay Card
/// =============================================================================
class TranslationOverlay extends StatelessWidget {
  final TranslationResult result;
  final double bufferProgress;
  final bool isBuffering;

  const TranslationOverlay({
    Key? key,
    required this.result,
    required this.bufferProgress,
    this.isBuffering = false,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final bool hasPrediction = result.action.isNotEmpty && result.action != '...';
    final Color accentColor = result.isConfident
        ? const Color(0xFF00E676) // Green
        : (hasPrediction ? const Color(0xFF00E5FF) : Colors.white38);

    return ClipRRect(
      borderRadius: BorderRadius.circular(24),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 12, sigmaY: 12),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          decoration: BoxDecoration(
            color: const Color(0xFF1E1E2E).withOpacity(0.85),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(
              color: accentColor.withOpacity(0.4),
              width: 1.5,
            ),
            boxShadow: [
              BoxShadow(
                color: accentColor.withOpacity(0.15),
                blurRadius: 20,
                spreadRadius: 2,
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header Row: Label & Latency
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 8,
                        height: 8,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: isBuffering ? Colors.orangeAccent : accentColor,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        isBuffering ? "BUFFERING SEQUENCE" : "PREDICTED GESTURE",
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.6),
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 1.2,
                        ),
                      ),
                    ],
                  ),
                  if (result.latencyMs > 0)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.08),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        "⚡ ${result.latencyMs.toStringAsFixed(0)} ms",
                        style: const TextStyle(
                          color: Color(0xFF00E5FF),
                          fontSize: 11,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                ],
              ),
              const SizedBox(height: 10),

              // Predicted Gesture Label
              Text(
                result.action.replaceAll('_', ' ').toUpperCase(),
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 28,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 1.5,
                  shadows: [
                    Shadow(
                      color: accentColor.withOpacity(0.6),
                      blurRadius: 12,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),

              // Confidence Progress Bar & Percentage
              Row(
                children: [
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: SizedBox(
                        height: 8,
                        child: LinearProgressIndicator(
                          value: isBuffering ? bufferProgress : result.confidence,
                          backgroundColor: Colors.white.withOpacity(0.1),
                          valueColor: AlwaysStoppedAnimation<Color>(
                            isBuffering ? Colors.orangeAccent : accentColor,
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    isBuffering
                        ? "${(bufferProgress * 100).toInt()}%"
                        : "${(result.confidence * 100).toStringAsFixed(1)}%",
                    style: TextStyle(
                      color: isBuffering ? Colors.orangeAccent : Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
