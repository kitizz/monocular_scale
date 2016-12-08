//
//  AVCaptureManager.m
//  SlowMotionVideoRecorder
//  https://github.com/shu223/SlowMotionVideoRecorder
//
//
//    The MIT License (MIT)
//
//    Copyright (c) 2013 shu223
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy of
//    this software and associated documentation files (the "Software"), to deal in
//    the Software without restriction, including without limitation the rights to
//    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//    the Software, and to permit persons to whom the Software is furnished to do so,
//    subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
//    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
//    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#import "AVCaptureManager.h"
#import <AVFoundation/AVFoundation.h>


@interface AVCaptureManager ()
<AVCaptureFileOutputRecordingDelegate>
{
    CMTime defaultVideoMaxFrameDuration;
    BOOL usingRearCam;
}
@property (nonatomic, strong) AVCaptureSession *captureSession;
@property (nonatomic, strong) AVCaptureMovieFileOutput *fileOutput;
@property (nonatomic, strong) AVCaptureDeviceFormat *defaultFormat;
@property (nonatomic, strong) AVCaptureVideoPreviewLayer *previewLayer;
@property (nonatomic, strong) AVCaptureStillImageOutput *imageOutput;
@property (nonatomic, strong) AVCaptureDevice *frontCam;
@property (nonatomic, strong) AVCaptureDevice *rearCam;
@property (nonatomic, strong) AVCaptureDeviceInput *currentInput;
@end


@implementation AVCaptureManager

- (id)initWithPreviewView:(UIView *)previewView {
    
    self = [super init];
    
    if (self) {
        
        NSError *error;
        
        self.captureSession = [[AVCaptureSession alloc] init];
        self.captureSession.sessionPreset = AVCaptureSessionPresetInputPriority;
        
        // Image Capture
        self.imageOutput = [[AVCaptureStillImageOutput alloc] init];
        
        // Allocate front a rear cameras
        NSArray *devices = [AVCaptureDevice devices];
        for (AVCaptureDevice *device in devices) {
            if ([device hasMediaType:AVMediaTypeVideo]) {
                if ([device position] == AVCaptureDevicePositionBack) {
                    //NSLog(@"Device position : back");
                    self.rearCam = device;
                }
                else if ([device position] == AVCaptureDevicePositionFront) {
                    //NSLog(@"Device position : front");
                    self.frontCam = device;
                }
            }
        }
        
        usingRearCam = true;
//        AVCaptureDevice *videoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        self.currentInput = [AVCaptureDeviceInput deviceInputWithDevice:self.rearCam error:&error];
        
        if (error) {
            NSLog(@"Video input creation failed");
            return nil;
        }
        
        if (![self.captureSession canAddInput:self.currentInput]) {
            NSLog(@"Video input add-to-session failed");
            return nil;
        }
        [self.captureSession addInput:self.currentInput];
        
        
        // save the default format
        [self switchFormatWithDesiredFPS:30.0];
        self.defaultFormat = self.rearCam.activeFormat;
        defaultVideoMaxFrameDuration = self.rearCam.activeVideoMaxFrameDuration;
        
        AVCaptureDevice *audioDevice= [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeAudio];
        AVCaptureDeviceInput *audioIn = [AVCaptureDeviceInput deviceInputWithDevice:audioDevice error:&error];
        [self.captureSession addInput:audioIn];
        
        self.fileOutput = [[AVCaptureMovieFileOutput alloc] init];
        [self.captureSession addOutput:self.fileOutput];
        [self.captureSession addOutput:self.imageOutput];
        
        // Force landscape orientation
        AVCaptureConnection *videoConnection = nil;
        
        for ( AVCaptureConnection *connection in [self.fileOutput connections] )
        {
            NSLog(@"%@", connection);
            for ( AVCaptureInputPort *port in [connection inputPorts] )
            {
                NSLog(@"%@", port);
                if ( [[port mediaType] isEqual:AVMediaTypeVideo] )
                {
                    videoConnection = connection;
                }
            }
        }
        
//        if([videoConnection isVideoOrientationSupported]) // **Here it is, its always false**
//        {
            [videoConnection setVideoOrientation: AVCaptureVideoOrientationLandscapeLeft];
//        }
        
        self.previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:self.captureSession];
        self.previewLayer.frame = previewView.bounds;
        self.previewLayer.contentsGravity = kCAGravityResizeAspectFill;
        self.previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
        [previewView.layer insertSublayer:self.previewLayer atIndex:0];
        
        [self.captureSession startRunning];
    }
    return self;
}



// =============================================================================
#pragma mark - Public

- (void)toggleContentsGravity {
    
    if ([self.previewLayer.videoGravity isEqualToString:AVLayerVideoGravityResizeAspectFill]) {
    
        self.previewLayer.videoGravity = AVLayerVideoGravityResizeAspect;
    }
    else {
        self.previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
    }
}

- (void)resetFormat {

    BOOL isRunning = self.captureSession.isRunning;
    
    if (isRunning) {
        [self.captureSession stopRunning];
    }

    AVCaptureDevice *videoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    [videoDevice lockForConfiguration:nil];
    videoDevice.activeFormat = self.defaultFormat;
    videoDevice.activeVideoMaxFrameDuration = defaultVideoMaxFrameDuration;
    [videoDevice unlockForConfiguration];

    if (isRunning) {
        [self.captureSession startRunning];
    }
}

- (void)switchFormatWithDesiredFPS:(CGFloat)desiredFPS
{
    BOOL isRunning = self.captureSession.isRunning;
    
    if (isRunning)  [self.captureSession stopRunning];
    
    // Switch to rear cam if necessary
    if (!usingRearCam) {
        [self.captureSession beginConfiguration];
        if (self.currentInput) {
            [self.captureSession removeInput:self.currentInput];
        }
        usingRearCam = true;
        NSError *error;
        self.currentInput = [AVCaptureDeviceInput deviceInputWithDevice:self.rearCam error:&error];
        
        if (!self.currentInput || error) {
            NSLog(@"WARNING: Unable to open Rear Camera: %@", error);
            [self.captureSession commitConfiguration];
            return;
        }
        [self.captureSession addInput:self.currentInput];
        [self.captureSession commitConfiguration];
    }
    
    AVCaptureDeviceFormat *selectedFormat = nil;
    int32_t maxWidth = 0;
    AVFrameRateRange *frameRateRange = nil;

    for (AVCaptureDeviceFormat *format in [self.rearCam formats]) {
        
        for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
            
            CMFormatDescriptionRef desc = format.formatDescription;
            CMVideoDimensions dimensions = CMVideoFormatDescriptionGetDimensions(desc);
            int32_t width = MIN(dimensions.width, dimensions.height);

            if (range.minFrameRate <= desiredFPS && desiredFPS <= range.maxFrameRate && width >= maxWidth && width <= 720) {
                
                selectedFormat = format;
                frameRateRange = range;
                maxWidth = width;
            }
        }
    }
    
    if (selectedFormat) {
        
        if ([self.rearCam lockForConfiguration:nil]) {

            NSLog(@"selected format:%@", selectedFormat);
            self.rearCam.activeFormat = selectedFormat;
            self.rearCam.activeVideoMinFrameDuration = CMTimeMake(1, (int32_t)desiredFPS);
            self.rearCam.activeVideoMaxFrameDuration = CMTimeMake(1, (int32_t)desiredFPS);
//            [videoDevice setExposureMode:AVCaptureExposureModeLocked];
            [self.rearCam unlockForConfiguration];
        }
    }
    
    if (isRunning) [self.captureSession startRunning];
}

- (void)switchToRearCam {
    if (usingRearCam) return;
    
    [self switchFormatWithDesiredFPS:120.0];
}

- (void)switchToFrontCam {
    if (!usingRearCam) return; // Already on front cam
    usingRearCam = false;
    
    BOOL isRunning = self.captureSession.isRunning;
    if (isRunning)  [self.captureSession stopRunning];
    
    [self.captureSession beginConfiguration];
    if (self.currentInput) {
        [self.captureSession removeInput:self.currentInput];
    }
    
    NSError *error;
    self.currentInput = [AVCaptureDeviceInput deviceInputWithDevice:self.frontCam error:&error];
    if (!self.currentInput || error) {
        NSLog(@"WARNING: Unable to open Front Facing camera: %@", error);
        [self.captureSession commitConfiguration];
        return;
    }
    
    [self.captureSession addInput:self.currentInput];
    [self.captureSession commitConfiguration];
    
    if (isRunning) [self.captureSession startRunning];
}

- (NSURL*)startRecording {
    AVCaptureDevice *device = self.currentInput.device;
    [device lockForConfiguration:nil];
    if ([device isFocusModeSupported: AVCaptureFocusModeLocked]) {
        device.focusMode = AVCaptureFocusModeLocked;
    }
    [device unlockForConfiguration];
    
    [self.fileOutput connectionWithMediaType:AVMediaTypeVideo].videoOrientation = AVCaptureVideoOrientationLandscapeLeft;
    
    NSURL *fileURL = [self getDatedFile:@"mp4"];
    [self.fileOutput startRecordingToOutputFileURL:fileURL recordingDelegate:self];

    return fileURL;
}

- (void)stopRecording {

    [self.fileOutput stopRecording];
}

- (BOOL)captureImage {
    AVCaptureConnection *videoConnection = [self.imageOutput connectionWithMediaType:AVMediaTypeVideo];;
    NSURL *fileURL = [self getDatedFile:@"jpg"];
    
    if (!videoConnection) {
        NSLog(@"WARNING: Unable to find videoConnection for image capture...");
        return false;
    }
    
    [self.imageOutput captureStillImageAsynchronouslyFromConnection:videoConnection completionHandler:
        ^(CMSampleBufferRef imageSampleBuffer, NSError *error) {

        if (!error) {
            NSData *imageData = [AVCaptureStillImageOutput jpegStillImageNSDataRepresentation:imageSampleBuffer];
            [[NSFileManager defaultManager] createFileAtPath:[fileURL path] contents:imageData attributes:nil];
        }
         
        if ([self.delegate respondsToSelector:@selector(didFinishSavingImageAtURL:error:)]) {
            [self.delegate didFinishSavingImageAtURL:fileURL error:error];
        }
     }];
    
    return true;
}


#pragma mark - Private
- (AVCaptureConnection*) getCaptureConnection {
//    AVCaptureConnection *videoConnection = nil;
    for (AVCaptureConnection *connection in self.imageOutput.connections) {
        for (AVCaptureInputPort *port in [connection inputPorts]) {
            if ([[port mediaType] isEqual:AVMediaTypeVideo] ) {
                return connection;
            }
        }
//        if (videoConnection) { break; }
    }
    return nil;
}

- (NSURL *) getDatedFile:(NSString *)ext {
    NSDateFormatter* formatter = [[NSDateFormatter alloc] init];
    [formatter setDateFormat:@"yyyy_MM_dd_HH_mm_ss"];
    NSString* dateTimePrefix = [formatter stringFromDate:[NSDate date]];
    
    int fileNamePostfix = 0;
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    
    NSString *tmpDir = [NSString stringWithFormat:@"%@/tmp", documentsDirectory];
    [[NSFileManager defaultManager] createDirectoryAtPath:tmpDir withIntermediateDirectories:YES attributes:nil error:nil];
    
    NSString *filePath = nil;
    NSString *fileName = nil;
    do {
        fileName = [NSString stringWithFormat:@"ipad_%@_%i", dateTimePrefix, fileNamePostfix++];
        filePath =[NSString stringWithFormat:@"%@/%@.%@", tmpDir, fileName, ext];
    } while ([[NSFileManager defaultManager] fileExistsAtPath:filePath]);
    
    return [NSURL URLWithString:[@"file://" stringByAppendingString:filePath]];
}

// =============================================================================
#pragma mark - AVCaptureFileOutputRecordingDelegate

- (void)                 captureOutput:(AVCaptureFileOutput *)captureOutput
    didStartRecordingToOutputFileAtURL:(NSURL *)fileURL
                       fromConnections:(NSArray *)connections
{
    _isRecording = YES;
}

- (void)                 captureOutput:(AVCaptureFileOutput *)captureOutput
   didFinishRecordingToOutputFileAtURL:(NSURL *)outputFileURL
                       fromConnections:(NSArray *)connections error:(NSError *)error
{
//    [self saveRecordedFile:outputFileURL];
    _isRecording = NO;
    
    if ([self.delegate respondsToSelector:@selector(didFinishRecordingToOutputFileAtURL:error:)]) {
        [self.delegate didFinishRecordingToOutputFileAtURL:outputFileURL error:error];
    }
    
    AVCaptureDevice *device = self.currentInput.device;
    [device lockForConfiguration:nil];
    if ([device isFocusModeSupported:AVCaptureFocusModeAutoFocus]) {
        device.focusMode = AVCaptureFocusModeContinuousAutoFocus;
    }
    [device unlockForConfiguration];
}

@end
