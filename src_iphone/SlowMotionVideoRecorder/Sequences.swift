//
//  Sequence.swift
//  Camera_IMU
//
// Copyright Simon Lucey 2015, All rights Reserved......

import Foundation
import UIKit

struct Sequence
{
    var name:String = ""
    
    init(name:String)
    {
        self.name = name
    }
}

@objc class Sequences : NSObject
{
    var sequences:[Sequence] = []
    var fileMgr: FileManager = FileManager.default
    var appDir: URL!
    
    var activeSequence: Int = -1
    var currentStep: Int = -1
    
    override init(){
        super.init()
        // Document directory access: http://stackoverflow.com/a/27722526
        self.appDir = self.fileMgr.urls(for: .documentDirectory, in: .userDomainMask)[0] 
        
        let f = indexFile()
        if (!self.fileMgr.fileExists(atPath: f)) {
            // Start a new index
            writeJsonIndex()
        } else {
            // Read in the existing index
            readJsonIndex()
        }
        debug()
        
        if let directoryContents =
                try? self.fileMgr.contentsOfDirectory(atPath: self.appDir.path) {
            print("Directories:")
            print(directoryContents)
        }
    }
    
    func indexFile() -> String {
        return (self.appDir.path as NSString).appendingPathComponent("index.json")
    }
    
    func debug() {
        print("Sequences:")
        print(sequences)
        for seq in sequences {
            print("\t\(seq.name)")
        }
    }
    
    func writeJsonIndex() {
        var names: [String] = []
        for seq in sequences {
//            names.append(seq.name)
            names.insert(seq.name, at: 0)
        }
        print("Writing JSON. Names: \(names)")
        let json = ["sequences": names, "active": activeSequence, "currentStep": currentStep ] as [String : Any]
        if let data = try? JSONSerialization.data(withJSONObject: json, options: .prettyPrinted) {
            self.fileMgr.createFile(atPath: indexFile(), contents:data, attributes:nil)
        }
    }
    
    func readJsonIndex() {
        var missingDir = false
        if let data = self.fileMgr.contents(atPath: indexFile()) {
            let json = JSON(data: data)
            print("Iterating")
            sequences.removeAll(keepingCapacity: true)
            for (_, name):(String, JSON) in json["sequences"] {
                if let namestr = name.string {
                    let dir = getDirFor(namestr)
                    if self.fileMgr.fileExists(atPath: dir) {
                        sequences.insert(Sequence(name: namestr), at: 0)
//                        sequences.append(Sequence(name: namestr))
                    } else {
                        missingDir = true
                    }
                }
            }
            if let activeVal = json["active"].int {
                activeSequence = activeVal
                if let currentVal = json["currentStep"].int {
                    currentStep = currentVal
                } else {
                    currentStep = 0
                }
            } else {
                activeSequence = -1
                currentStep = -1
            }
        }
        
        if missingDir {
            // Sequence directories have been removed. Update the index
            writeJsonIndex()
        }
    }
    
    func beginRecording(_ name: String) -> Bool {
        for (_, seq) in self.sequences.enumerated() {
            if seq.name == name {
                print("Sequence with name \(name) already exists!")
                return false
            }
        }
        
        let newDir = (self.appDir.path as NSString).appendingPathComponent(name)
        do {
            try self.fileMgr.createDirectory(atPath: newDir, withIntermediateDirectories: true, attributes: nil)
        } catch _ {
        }
        sequences.insert(Sequence(name: name), at: 0)
        
        activeSequence = 0
        currentStep = 0
        writeJsonIndex()
        
        return true
    }
    
    func changeActiveSequence(_ index: Int) -> Bool {
        if index < 0 || index >= sequences.count {
            return false
        }
        activeSequence = index
        currentStep = 0
        writeJsonIndex()
        return true
    }
    
    func imageForSequence(_ index: Int) -> UIImage? {
        let imagePath = (getDirFor(sequences[index].name) as NSString).appendingPathComponent("thumbnail.jpg")
        if let data = try? Data(contentsOf: URL(fileURLWithPath: imagePath)) {
            return UIImage(data: data)
        }
        return nil;
    }
    
    func activeName() -> String {
        if activeSequence < 0 || activeSequence >= sequences.count {
            return ""
        }
        return sequences[activeSequence].name
    }
    
    func lastStep() {
        if currentStep > 0 {
            currentStep -= 1
            writeJsonIndex()
        }
    }
    func nextStep() {
        currentStep += 1
        writeJsonIndex()
    }
    
    func getActiveDir() -> String {
        if activeSequence < 0 { return "" }
        
        return getDirFor(sequences[activeSequence].name)
    }
    
    func getDirFor(_ name: String) -> String {
        return (self.appDir.path as NSString).appendingPathComponent(name)
    }

    func setScanForActiveSequence(_ url: URL!) {
        saveActiveFile(url, to: "scan.mp4")
    }
    
    func setPortraitForActiveSequence(_ url: URL!) {
        if activeSequence < 0 { return }
        
        let imagePath = (getActiveDir() as NSString).appendingPathComponent("portrait.jpg")
        let thumbPath = (getActiveDir() as NSString).appendingPathComponent("thumbnail.jpg")
        
        if let srcPath = url?.path {
            let imageData: Data? = self.fileMgr.contents(atPath: srcPath)
            if imageData == nil {
                print("WARNING: Cannot read image from \(srcPath)")
                return
            }
            
            do {
                try self.fileMgr.removeItem(atPath: srcPath)
            } catch _ {
            }
            
            let success = self.fileMgr.createFile(atPath: imagePath, contents:imageData, attributes:nil)
            if !success {
                print("WARNING: Unable to move portrait image \(srcPath) to \(imagePath)")
                return
            } else {
                print("Successfully moved portrait image \(srcPath) to \(imagePath)")
            }
            
            print("Attempting to make thumbnail...")
            if let image = UIImage(data: imageData!) {
                let ratio = image.size.height/image.size.width
                let size: CGSize = CGSize(width: 128, height: Int(128*ratio))
                UIGraphicsBeginImageContext(size)
                image.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
                let thumbnail = UIGraphicsGetImageFromCurrentImageContext()
                try? UIImageJPEGRepresentation(thumbnail!, 0.98)!.write(to: URL(fileURLWithPath: thumbPath), options: [.atomic])
            } else {
                print("WARNING: Unable to create UIImage from data...")
            }
        }
    }
    
    func setScaleQRForActiveSequence(_ url: URL!) {
        saveActiveFile(url, to: "qr.jpg")
    }
    
    func setScaleVideoForActiveSequence(_ url: URL!) {
        saveActiveFile(url, to: "imu.mp4")
    }
    
    func setIMULogForActiveSequence(_ imuLogUrl: URL!) {
        saveActiveFile(imuLogUrl, to: "imu.txt")
    }
    
    func saveActiveFile(_ from: URL, to: String) {
        if activeSequence < 0 { return }
        
        let videoPath = (getActiveDir() as NSString).appendingPathComponent(to)
        
        let srcPath = from.path
        if !self.fileMgr.fileExists(atPath: srcPath) {
            print("Source doesn't exist: \(srcPath)")
            return
        }
        
        do {
            // moveItemAtPath doesn't allow overwriting
            try self.fileMgr.removeItem(atPath: videoPath)
        } catch _ {
        }
        let success: Bool
        do {
            try self.fileMgr.moveItem(atPath: srcPath, toPath: videoPath)
            success = true
        } catch _ {
            success = false
        }
        if !success {
            print("WARNING: Unable to move file \(srcPath) to \(videoPath)")
        } else {
            print("Successfully moved file \(srcPath) to \(videoPath)")
        }
    }
    
    func add(_ name: String) {
        print("Adding sequence: \(name)...")
        for (index, seq) in self.sequences.enumerated() {
            if seq.name == name {
                print("Sequence with name \(name) already exists!")
                return
            }
        }
        let newDir = (self.appDir.path as NSString).appendingPathComponent(name)
        do {
            try self.fileMgr.createDirectory(atPath: newDir, withIntermediateDirectories: true, attributes: nil)
        } catch _ {
        }
        sequences.insert(Sequence(name: name), at: 0)
        debug()
        // Update the index file
        writeJsonIndex()
    }
    
    func remove(_ name: String) {
        print("Deleting \(name)")
        // Delete folder?
        let newDir = (self.appDir.path as NSString).appendingPathComponent(name)
        let success: Bool
        do {
            try self.fileMgr.removeItem(atPath: newDir)
            success = true
        } catch _ {
            success = false
        }
        print("Removing \(newDir). Success: \(success)")

        for (index, seq) in self.sequences.enumerated() {
            if seq.name == name {
                print("Found \(name). Deleting...")
                sequences.remove(at: index)
                
                // Update the index file
                writeJsonIndex()
                return
            }
        }
    }
    
    func removeIndex(_ index: Int) {
        let name = sequences[index].name
        print("Deleting \(name)")
        // Delete folder?
        let newDir = (self.appDir.path as NSString).appendingPathComponent(name)
        let success: Bool
        do {
            try self.fileMgr.removeItem(atPath: newDir)
            success = true
        } catch _ {
            success = false
        }
        print("Removing \(newDir). Success: \(success)")

        sequences.remove(at: index)
        
        // Update the index file
        writeJsonIndex()
    }
    
    func at(_ index: Int) -> Sequence {
        return sequences[index]
    }
    
    func count() -> Int {
        return sequences.count
    }
}
