import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// =============================================================================
/// Project Vani: Accumulated Sentence History Builder Widget
/// =============================================================================
class SentenceBar extends StatelessWidget {
  final List<String> sentence;
  final VoidCallback onClear;

  const SentenceBar({
    Key? key,
    required this.sentence,
    required this.onClear,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final String fullText = sentence.join(" ");

    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: const Color(0xFF161622).withOpacity(0.85),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: Colors.white.withOpacity(0.12),
              width: 1,
            ),
          ),
          child: Row(
            children: [
              const Icon(
                Icons.translate_rounded,
                color: Color(0xFF00E5FF),
                size: 22,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  sentence.isEmpty
                      ? "Perform gestures in front of the camera..."
                      : fullText.toUpperCase(),
                  style: TextStyle(
                    color: sentence.isEmpty ? Colors.white38 : Colors.white,
                    fontSize: 15,
                    fontWeight: sentence.isEmpty ? FontWeight.normal : FontWeight.w600,
                    letterSpacing: 0.8,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if (sentence.isNotEmpty) ...[
                IconButton(
                  icon: const Icon(Icons.copy_rounded, color: Colors.white70, size: 18),
                  tooltip: "Copy Text",
                  onPressed: () {
                    Clipboard.setData(ClipboardData(text: fullText));
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text("Translation copied to clipboard!"),
                        duration: Duration(seconds: 1),
                        behavior: SnackBarBehavior.floating,
                      ),
                    );
                  },
                ),
                IconButton(
                  icon: const Icon(Icons.clear_all_rounded, color: Colors.white70, size: 20),
                  tooltip: "Clear Translation",
                  onPressed: onClear,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
