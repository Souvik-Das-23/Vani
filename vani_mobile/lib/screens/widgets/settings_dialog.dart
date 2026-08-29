import 'package:flutter/material.dart';
import '../../config/app_config.dart';
import '../../services/api_service.dart';

/// =============================================================================
/// Project Vani: Backend Server Configuration Dialog
/// =============================================================================
class BackendSettingsDialog extends StatefulWidget {
  final VoidCallback onConfigSaved;

  const BackendSettingsDialog({Key? key, required this.onConfigSaved})
      : super(key: key);

  @override
  State<BackendSettingsDialog> createState() => _BackendSettingsDialogState();
}

class _BackendSettingsDialogState extends State<BackendSettingsDialog> {
  late final TextEditingController _urlController;
  final ApiService _apiService = ApiService();
  bool _isTesting = false;
  String? _testResult;
  bool _isHealthy = false;

  @override
  void initState() {
    super.initState();
    _urlController = TextEditingController(text: AppConfig.baseUrl);
  }

  @override
  void dispose() {
    _urlController.dispose();
    _apiService.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    setState(() {
      _isTesting = true;
      _testResult = null;
    });

    final previousUrl = AppConfig.baseUrl;
    AppConfig.baseUrl = _urlController.text.trim();

    final healthy = await _apiService.checkHealth();

    setState(() {
      _isTesting = false;
      _isHealthy = healthy;
      _testResult = healthy
          ? "Connected! Django server is healthy."
          : "Connection failed. Check server IP & port.";
    });

    if (!healthy) {
      AppConfig.baseUrl = previousUrl;
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: const Color(0xFF1E1E2E),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      title: Row(
        children: const [
          Icon(Icons.settings_ethernet, color: Color(0xFF00E5FF)),
          SizedBox(width: 10),
          Text(
            "Backend Settings",
            style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
          ),
        ],
      ),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Django Inference API Base URL:",
              style: TextStyle(color: Colors.white70, fontSize: 13),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _urlController,
              style: const TextStyle(color: Colors.white, fontSize: 14),
              decoration: InputDecoration(
                filled: true,
                fillColor: const Color(0xFF282A3A),
                hintText: "http://10.0.2.2:8000/api",
                hintStyle: const TextStyle(color: Colors.white30),
                contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
                prefixIcon: const Icon(Icons.link, color: Color(0xFF00E5FF), size: 20),
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 6,
              children: [
                _buildPresetChip("Android Emulator", "http://10.0.2.2:8000/api"),
                _buildPresetChip("Localhost / Web", "http://127.0.0.1:8000/api"),
              ],
            ),
            const SizedBox(height: 16),
            if (_testResult != null)
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: _isHealthy
                      ? const Color(0xFF00E676).withOpacity(0.15)
                      : const Color(0xFFFF5252).withOpacity(0.15),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: _isHealthy ? const Color(0xFF00E676) : const Color(0xFFFF5252),
                    width: 1,
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      _isHealthy ? Icons.check_circle : Icons.error,
                      color: _isHealthy ? const Color(0xFF00E676) : const Color(0xFFFF5252),
                      size: 18,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _testResult!,
                        style: TextStyle(
                          color: _isHealthy ? const Color(0xFF00E676) : const Color(0xFFFF5252),
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: _isTesting ? null : _testConnection,
          child: _isTesting
              ? const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF00E5FF)),
                )
              : const Text("Test Connection", style: TextStyle(color: Color(0xFF00E5FF))),
        ),
        ElevatedButton(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF00E5FF),
            foregroundColor: Colors.black,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
          onPressed: () {
            AppConfig.baseUrl = _urlController.text.trim();
            widget.onConfigSaved();
            Navigator.of(context).pop();
          },
          child: const Text("Save", style: TextStyle(fontWeight: FontWeight.bold)),
        ),
      ],
    );
  }

  Widget _buildPresetChip(String label, String url) {
    return ActionChip(
      backgroundColor: const Color(0xFF282A3A),
      label: Text(label, style: const TextStyle(color: Colors.white70, fontSize: 11)),
      onPressed: () {
        _urlController.text = url;
      },
    );
  }
}
