// See https://aka.ms/new-console-template for more information
using System.IO.Abstractions;

var ocr = new DdddOCR.NET.Core.OcrReader();

var fileSystem = new FileSystem();
var images = fileSystem.Directory.GetFiles("test-images", "*.jpg", SearchOption.AllDirectories);
foreach(var image in images) {
    var result = ocr.ReadText(image);
    Console.WriteLine($"Image: {image} - Result: {result?.Substring(0, Math.Min(result.Length, 20))}...");
}
