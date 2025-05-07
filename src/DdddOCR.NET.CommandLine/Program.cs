using System.Diagnostics;
using System.IO.Abstractions;
using DdddOCR.NET;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;

var fileSystem = new FileSystem();
using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole(options => {
    options.FormatterName = ConsoleFormatterNames.Systemd;
}));
var ocr = new OcrReader(fileSystem, loggerFactory.CreateLogger<OcrReader>());

var consoleLogger = loggerFactory.CreateLogger<Program>();

var cts = new CancellationTokenSource();
Console.CancelKeyPress += (s, e) => {
    consoleLogger.LogInformation("Canceling...");
    cts.Cancel();
    e.Cancel = true;
};

var images = fileSystem.Directory.GetFiles("test-images", "*.jpg", SearchOption.AllDirectories);
int total = 0, failures = 0;
foreach(var image in images) {
    var stopwatch = Stopwatch.StartNew();
    var result = await ocr.ReadTextAsync(image, cts.Token);
    stopwatch.Stop();

    var fileName = fileSystem.Path.GetFileName(image);
    var isMatch = fileSystem.Path.GetFileNameWithoutExtension(image).Equals(result, StringComparison.OrdinalIgnoreCase);

    total++;
    if(!isMatch) {
        failures++;
    }

    consoleLogger.LogInformation("Image: {Image} - IsMatch: {IsMatch} - Result: {Result} - Time: {ElapsedMilliseconds} ms", image, isMatch, result, stopwatch.ElapsedMilliseconds);
}

var successRate = (double)(total - failures) / total * 100;
consoleLogger.LogInformation("Success Rate: {SuccessRate:F2}%", successRate);
