// vision_instance_masks.swift
//
// Extract foreground *instance* masks from each specimen scan with Apple Vision's
// VNGenerateForegroundInstanceMaskRequest, and write one PNG mask per instance.
// This runs ONLY on macOS 14+ (the request is unavailable on Linux/iOS-sim without
// a GPU). Build and run on a Mac:
//
//   swiftc -O vision_instance_masks.swift -o vision_instance_masks
//   ./vision_instance_masks shimahotarubukuro/ out_masks/
//
// For every image in the input dir it writes out_masks/<stem>/inst_<k>.png, an
// 8-bit mask (255 = instance, 0 = background) at the source resolution. The Python
// side then matches each instance to a reviewed corolla by mask overlap and uses it
// as the ROI (see integration notes in the chat).

import Foundation
import Vision
import CoreImage
import AppKit

let ci = CIContext()

func saveMask(_ pixelBuffer: CVPixelBuffer, to url: URL) {
    let image = CIImage(cvPixelBuffer: pixelBuffer)
    // Vision mask is a 1-channel float [0,1]; scale to 8-bit grayscale.
    guard let cg = ci.createCGImage(image, from: image.extent) else { return }
    let rep = NSBitmapImageRep(cgImage: cg)
    guard let png = rep.representation(using: .png, properties: [:]) else { return }
    try? png.write(to: url)
}

func process(_ imageURL: URL, _ outDir: URL) {
    let handler = VNImageRequestHandler(url: imageURL, options: [:])
    let request = VNGenerateForegroundInstanceMaskRequest()
    do {
        try handler.perform([request])
    } catch {
        FileHandle.standardError.write("perform failed for \(imageURL.lastPathComponent): \(error)\n".data(using: .utf8)!)
        return
    }
    guard let result = request.results?.first else {
        print("\(imageURL.lastPathComponent): no foreground instances")
        return
    }
    let stem = imageURL.deletingPathExtension().lastPathComponent
    let dir = outDir.appendingPathComponent(stem)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    var k = 0
    for instance in result.allInstances {
        do {
            // One instance at a time -> a clean per-object mask at source scale.
            let mask = try result.generateScaledMaskForImage(forInstances: [instance], from: handler)
            saveMask(mask, to: dir.appendingPathComponent(String(format: "inst_%02d.png", k)))
            k += 1
        } catch {
            FileHandle.standardError.write("instance \(instance) failed: \(error)\n".data(using: .utf8)!)
        }
    }
    print("\(stem): \(k) instance masks")
}

let args = CommandLine.arguments
guard args.count == 3 else {
    print("usage: vision_instance_masks <input_dir> <output_dir>")
    exit(2)
}
let inDir = URL(fileURLWithPath: args[1])
let outDir = URL(fileURLWithPath: args[2])
let exts: Set<String> = ["jpg", "jpeg", "png", "tif", "tiff"]
let files = (try? FileManager.default.contentsOfDirectory(at: inDir, includingPropertiesForKeys: nil)) ?? []
for f in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
    where exts.contains(f.pathExtension.lowercased()) {
    process(f, outDir)
}
