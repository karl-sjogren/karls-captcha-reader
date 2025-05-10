namespace Karls.CaptchaReader;

/// <summary>
/// Karls.CaptchaReader is a library for Optical Character Recognition (OCR).
/// It uses the ONNX runtime for inference and is designed to be fast and efficient.
/// </summary>
public interface IOcrReader {
    /// <summary>
    /// Reads text from an image file.
    /// </summary>
    Task<string?> ReadTextAsync(string imagePath, CancellationToken cancellationToken);

    /// <summary>
    /// Reads text from a byte array representing an image.
    /// </summary>
    Task<string?> ReadTextAsync(byte[] imageBytes, CancellationToken cancellationToken);
}
