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
    var fileMgr: NSFileManager = NSFileManager.defaultManager()
    var appDir: NSURL = NSURL()
    
    var activeSequence: Int = -1
    var currentStep: Int = -1
    
    override init(){
        super.init()
        // Document directory access: http://stackoverflow.com/a/27722526
        self.appDir = self.fileMgr.URLsForDirectory(.DocumentDirectory, inDomains: .UserDomainMask)[0] 
        
        let f = indexFile()
        if (!self.fileMgr.fileExistsAtPath(f)) {
            // Start a new index
            writeJsonIndex()
        } else {
            // Read in the existing index
            readJsonIndex()
        }
        debug()
        
        if let directoryContents =
            try? self.fileMgr.contentsOfDirectoryAtPath(self.appDir.path!) {
                print(directoryContents)
        }
    }
    
    func indexFile() -> String {
        return (self.appDir.path! as NSString).stringByAppendingPathComponent("index.json")
    }
    
    func debug() {
        print("Sequences:")
        for seq in sequences {
            print("\t\(seq.name)")
        }
    }
    
    func writeJsonIndex() {
        var names: [String] = []
        for seq in sequences {
            names.append(seq.name)
        }
        print("Writing JSON. Names: \(names)")
        let json = ["sequences": names, "active": activeSequence, "currentStep": currentStep ]
        if let data = try? NSJSONSerialization.dataWithJSONObject(json, options: .PrettyPrinted) {
            self.fileMgr.createFileAtPath(indexFile(), contents:data, attributes:nil)
        }
    }
    
    func readJsonIndex() {
        var missingDir = false
        if let data = self.fileMgr.contentsAtPath(indexFile()) {
            if let json = (try? NSJSONSerialization.JSONObjectWithData(data, options: .MutableContainers)) as! NSDictionary? {
                sequences.removeAll(keepCapacity: true)
                for name in (json["sequences"] as! [String]) {
                    let dir = getDirFor(name)
                    if self.fileMgr.fileExistsAtPath(dir) {
                        sequences.append(Sequence(name: name))
                    } else {
                        missingDir = true
                    }
                }
                if let activeVal: AnyObject = json["active"] {
                    activeSequence = activeVal as! Int
                    if let currentVal: AnyObject = json["currentStep"] {
                        currentStep = currentVal as! Int
                    } else {
                        currentStep = 0
                    }
                } else {
                    activeSequence = -1
                    currentStep = -1
                }
            }
        }
        
        if missingDir {
            // Sequence directories have been removed. Update the index
            writeJsonIndex()
        }
    }
    
    func beginRecording(name: String) -> Bool {
        for (index, seq) in self.sequences.enumerate() {
            if seq.name == name {
                print("Sequence with name \(name) already exists!")
                return false
            }
        }
        
        let newDir = (self.appDir.path! as NSString).stringByAppendingPathComponent(name)
        do {
            try self.fileMgr.createDirectoryAtPath(newDir, withIntermediateDirectories: true, attributes: nil)
        } catch _ {
        }
        sequences.append(Sequence(name: name))
        
        activeSequence = sequences.count - 1
        currentStep = 0
        writeJsonIndex()
        
        return true
    }
    
    func changeActiveSequence(index: Int) -> Bool {
        if index < 0 || index >= sequences.count {
            return false
        }
        activeSequence = index
        currentStep = 0
        writeJsonIndex()
        return true
    }
    
    func imageForSequence(index: Int) -> UIImage? {
        let imagePath = (getDirFor(sequences[index].name) as NSString).stringByAppendingPathComponent("thumbnail.jpg")
        if let data = NSData(contentsOfFile: imagePath) {
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
    
    func getDirFor(name: String) -> String {
        return (self.appDir.path! as NSString).stringByAppendingPathComponent(name)
    }

    func setScanForActiveSequence(url: NSURL!) {
        saveActiveFile(url, to: "scan.mp4")
    }
    
    func setPortraitForActiveSequence(url: NSURL!) {
        if activeSequence < 0 { return }
        
        let imagePath = (getActiveDir() as NSString).stringByAppendingPathComponent("portrait.jpg")
        let thumbPath = (getActiveDir() as NSString).stringByAppendingPathComponent("thumbnail.jpg")
        
        if let srcPath = url.path {
            let imageData: NSData? = self.fileMgr.contentsAtPath(srcPath)
            if imageData == nil {
                print("WARNING: Cannot read image from \(srcPath)")
                return
            }
            
            do {
                try self.fileMgr.removeItemAtPath(srcPath)
            } catch _ {
            }
            
            let success = self.fileMgr.createFileAtPath(imagePath, contents:imageData, attributes:nil)
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
                image.drawInRect(CGRectMake(0, 0, size.width, size.height))
                let thumbnail = UIGraphicsGetImageFromCurrentImageContext()
                UIImageJPEGRepresentation(thumbnail!, 0.98)!.writeToFile(thumbPath, atomically: true)
            } else {
                print("WARNING: Unable to create UIImage from data...")
            }
        }
    }
    
    func setScaleQRForActiveSequence(url: NSURL!) {
        saveActiveFile(url, to: "qr.jpg")
    }
    
    func setScaleVideoForActiveSequence(url: NSURL!) {
        saveActiveFile(url, to: "imu.mp4")
    }
    
    func setIMULogForActiveSequence(imuLogUrl: NSURL!) {
        saveActiveFile(imuLogUrl, to: "imu.txt")
    }
    
    func saveActiveFile(from: NSURL, to: String) {
        if activeSequence < 0 { return }
        
        let videoPath = (getActiveDir() as NSString).stringByAppendingPathComponent(to)
        
        if let srcPath = from.path {
            if !self.fileMgr.fileExistsAtPath(srcPath) {
                print("Source doesn't exist: \(srcPath)")
                return
            }
            
            do {
                // moveItemAtPath doesn't allow overwriting
                try self.fileMgr.removeItemAtPath(videoPath)
            } catch _ {
            }
            let success: Bool
            do {
                try self.fileMgr.moveItemAtPath(srcPath, toPath: videoPath)
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
    }
    
    func add(name: String) {
        print("Adding sequence: \(name)...")
        for (index, seq) in self.sequences.enumerate() {
            if seq.name == name {
                print("Sequence with name \(name) already exists!")
                return
            }
        }
        let newDir = (self.appDir.path! as NSString).stringByAppendingPathComponent(name)
        do {
            try self.fileMgr.createDirectoryAtPath(newDir, withIntermediateDirectories: true, attributes: nil)
        } catch _ {
        }
        sequences.append(Sequence(name: name))
        debug()
        // Update the index file
        writeJsonIndex()
    }
    
    func remove(name: String) {
        print("Deleting \(name)")
        // Delete folder?
        let newDir = (self.appDir.path! as NSString).stringByAppendingPathComponent(name)
        let success: Bool
        do {
            try self.fileMgr.removeItemAtPath(newDir)
            success = true
        } catch _ {
            success = false
        }
        print("Removing \(newDir). Success: \(success)")

        for (index, seq) in self.sequences.enumerate() {
            if seq.name == name {
                print("Found \(name). Deleting...")
                sequences.removeAtIndex(index)
                
                // Update the index file
                writeJsonIndex()
                return
            }
        }
    }
    
    func removeIndex(index: Int) {
        let name = sequences[index].name
        print("Deleting \(name)")
        // Delete folder?
        let newDir = (self.appDir.path! as NSString).stringByAppendingPathComponent(name)
        let success: Bool
        do {
            try self.fileMgr.removeItemAtPath(newDir)
            success = true
        } catch _ {
            success = false
        }
        print("Removing \(newDir). Success: \(success)")

        sequences.removeAtIndex(index)
        
        // Update the index file
        writeJsonIndex()
    }
    
    func at(index: Int) -> Sequence {
        return sequences[index]
    }
    
    func count() -> Int {
        return sequences.count
    }
}
