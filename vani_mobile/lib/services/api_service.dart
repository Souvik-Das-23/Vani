import 'dart:convert';
import 'dart:developer' as developer;
import 'package:http/http.dart' as http;
import '../config/app_config.dart';
import '../models/translation_result.dart';

/// =============================================================================
/// Project Vani: Django Backend REST API Service
/// =============================================================================
class ApiService {
  final http.Client _client;

  ApiService({http.Client? client}) : _client = client ?? http.Client();

  /// Sends the 30-frame temporal coordinate matrix to the Django inference backend
  Future<TranslationResult> translateSequence(List<List<double>> sequence) async {
    final uri = Uri.parse(AppConfig.translateUrl);
    final stopwatch = Stopwatch()..start();

    try {
      final payload = jsonEncode({'sequence': sequence});

      final response = await _client
          .post(
            uri,
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            body: payload,
          )
          .timeout(const Duration(seconds: 4));

      stopwatch.stop();

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return TranslationResult.fromJson(data);
      } else {
        developer.log(
          'API Error ${response.statusCode}: ${response.body}',
          name: 'ApiService',
        );
        return TranslationResult(
          status: 'error',
          action: 'Error (${response.statusCode})',
          confidence: 0.0,
          probabilities: {},
          sequenceLength: sequence.length,
          latencyMs: stopwatch.elapsedMilliseconds.toDouble(),
        );
      }
    } catch (e) {
      developer.log('Translation request failed: $e', name: 'ApiService');
      return TranslationResult(
        status: 'error',
        action: 'Offline',
        confidence: 0.0,
        probabilities: {},
        sequenceLength: sequence.length,
        latencyMs: stopwatch.elapsedMilliseconds.toDouble(),
      );
    }
  }

  /// Verifies connectivity and health status of the Django server
  Future<bool> checkHealth() async {
    final uri = Uri.parse(AppConfig.healthUrl);
    try {
      final response = await _client
          .get(uri)
          .timeout(const Duration(seconds: 3));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return data['status'] == 'healthy';
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  /// Fetches the registered gesture action list from the server
  Future<List<String>> fetchActions() async {
    final uri = Uri.parse(AppConfig.actionsUrl);
    try {
      final response = await _client
          .get(uri)
          .timeout(const Duration(seconds: 3));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final actions = (data['actions'] as List<dynamic>?)
            ?.map((e) => e.toString())
            .toList();
        return actions ?? [];
      }
      return [];
    } catch (_) {
      return [];
    }
  }

  void dispose() {
    _client.close();
  }
}
