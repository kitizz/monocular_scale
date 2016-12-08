//
//  ViewController.m
//
// Copyright Simon Lucey 2015, All rights Reserved......

#import "ViewController.h"
#import "AVCaptureManager.h"
#import <AssetsLibrary/AssetsLibrary.h>
#import <CoreMotion/CoreMotion.h>

#import "Camera_IMU-Swift.h"


@interface ViewController ()
<AVCaptureManagerDelegate, UITextFieldDelegate>
{
    NSTimeInterval startTime;
    BOOL isNeededToSave;
    BOOL logOpen;
    NSMutableArray *faces;
    int step;
}
@property (nonatomic, strong) AVCaptureManager *captureManager;
@property (nonatomic, assign) NSTimer *timer;
@property (nonatomic, strong) UIImage *recStartImage;
@property (nonatomic, strong) UIImage *recStopImage;
@property (nonatomic, strong) UIImage *outerImage1;
@property (nonatomic, strong) UIImage *outerImage2;

@property (nonatomic, weak) IBOutlet UILabel *statusLabel;
@property (nonatomic, weak) IBOutlet UIButton *recBtn;
@property (nonatomic, weak) IBOutlet UIImageView *outerImageView;
@property (weak, nonatomic) IBOutlet UIButton *lastBtn;
@property (weak, nonatomic) IBOutlet UIButton *nextBtn;
@property (weak, nonatomic) IBOutlet UIBarButtonItem *sequencesBtn;
@property (weak, nonatomic) IBOutlet UIImageView *fiducialImg;

@property (strong, nonatomic) CMMotionManager *motionManager;
@property (strong, nonatomic) NSString *fileName;
@property (strong, nonatomic) NSString *appDir;
@property (strong, nonatomic) NSString *logPath;
@property (strong, nonatomic) NSString *camPath;
@property (strong, nonatomic) NSFileHandle *logFile;
@property (nonatomic, strong) Sequences *sequences;

@end


@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    self.captureManager = [[AVCaptureManager alloc] initWithPreviewView:self.view];
    self.captureManager.delegate = self;
    [self.captureManager switchToFrontCam];
    
    UITapGestureRecognizer *tapGesture = [[UITapGestureRecognizer alloc] initWithTarget:self
                                                                                 action:@selector(handleDoubleTap:)];
    tapGesture.numberOfTapsRequired = 2;
    [self.view addGestureRecognizer:tapGesture];
    
    self.sequences = [[Sequences alloc] init];
    
    // Set up IMU log file
    self.motionManager = [[CMMotionManager alloc] init];
    self.motionManager.deviceMotionUpdateInterval = 0.001;
    
    [self.motionManager startDeviceMotionUpdatesToQueue:[NSOperationQueue currentQueue]
        withHandler:^(CMDeviceMotion  *data, NSError *error) {
            CMAttitude *att = data.attitude;
            CMAcceleration grav = data.gravity;
            CMAcceleration acc = data.userAcceleration;
            CMRotationRate gyro = data.rotationRate;
            
            unsigned long long timestamp = [[NSNumber numberWithDouble:data.timestamp*1000000000] unsignedLongLongValue];;
            
            [self logAccelerationData:acc gravity:grav atTime:timestamp];
            [self logGyroscopeData:gyro atTime:timestamp];
            [self logAttitiudeData:att atTime:timestamp];
            
            if(error) NSLog(@"%@", error);
        }
     ];

    self.appDir = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
    self.camPath = [self.appDir stringByAppendingPathComponent:@"calibration.cam"];
    
    // Set up images for the Shutter Button
    UIImage *image;
    image = [UIImage imageNamed:@"ShutterButtonStart"];
    self.recStartImage = [image imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    [self.recBtn setImage:self.recStartImage
                 forState:UIControlStateNormal];
    
    image = [UIImage imageNamed:@"ShutterButtonStop"];
    self.recStopImage = [image imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];

    [self.recBtn setTintColor:[UIColor colorWithRed:245./255.
                                              green:51./255.
                                               blue:51./255.
                                              alpha:1.0]];
    self.outerImage1 = [UIImage imageNamed:@"outer1"];
    self.outerImage2 = [UIImage imageNamed:@"outer2"];
    self.outerImageView.image = self.outerImage1;
    
    // Set up the state for the Last and Next buttons
    [self.lastBtn setTitleColor:[UIColor grayColor] forState:UIControlStateDisabled];
    [self.nextBtn setTitleColor:[UIColor grayColor] forState:UIControlStateDisabled];
    
    [self loadState];
}

- (void)viewWillAppear:(BOOL)animated
{
    [self.navigationController.navigationItem setHidesBackButton:YES];
    [self.navigationController setNavigationBarHidden:NO animated:YES];
    [self.navigationController setToolbarHidden:YES animated:YES];
    [self loadState];
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
}


// =============================================================================
#pragma mark - Gesture Handler

- (void)handleDoubleTap:(UITapGestureRecognizer *)sender {

    [self.captureManager toggleContentsGravity];
}


// =============================================================================
#pragma mark - Private

- (void)loadState {
    NSInteger state = [self.sequences currentStep];
    
    // Could use intermediate vars to stop fast switching
    self.lastBtn.enabled = true;
    self.nextBtn.enabled = true;
    self.fiducialImg.hidden = true;
    [self.navigationController setTitle:[self.sequences activeName]];

    if (state == 0) {
        // Take profile shot
        self.statusLabel.text = @"Step 1: Take Portrait Shot";
        self.lastBtn.enabled = false;
        [self.captureManager switchToRearCam];

//    } else if (state == 1) {
//        // Take scan
//        self.statusLabel.text = @"Step 2: Record Scan";
//        [self.captureManager switchToFrontCam];
        
//    } else if (state == 2) {
//        // Take QR Code shot
//        self.statusLabel.text = @"Step 3: Take QR Shot";
//        [self.captureManager switchToFrontCam];
        
    } else if (state >= 1) {
        // Take IMU video
        self.statusLabel.text = @"Step 2: Record IMU Video";
        self.nextBtn.enabled = false;
        [self.captureManager switchToRearCam];
        
    }
}


- (void)saveRecordedFile:(NSURL *)recordedFile {
    [self.sequences setScanForActiveSequence:recordedFile];
}

// ===========
// IMU Methods
// ===========
-(void)logAccelerationData:(CMAcceleration)acc gravity:(CMAcceleration)grav atTime:(unsigned long long)t
{
    if (!self.captureManager.isRecording) return;
    NSString *accLine = [NSString stringWithFormat:@"LinearAcceleration,%f,%f,%f,%llu\n", 9.81*acc.x, 9.81*acc.y, 9.81*acc.z, t];
    NSString *gravLine = [NSString stringWithFormat:@"Gravity,%f,%f,%f,%llu\n", 9.81*grav.x, 9.81*grav.y, 9.81*grav.z, t];
    NSString *linAccLine = [NSString stringWithFormat:@"Accelerometer,%f,%f,%f,%llu\n", 9.81*(acc.x+grav.x), 9.81*(acc.y+grav.y), 9.81*(acc.z+grav.z), t];
    
    [self writeToLog:accLine];
    [self writeToLog:gravLine];
    [self writeToLog:linAccLine];
}

-(void)logAttitiudeData:(CMAttitude*)att atTime:(unsigned long long)t
{
    if (!self.captureManager.isRecording) return;
    CMQuaternion quat = att.quaternion;
    NSString *line = [NSString stringWithFormat:@"RotationVector,%f,%f,%f,%f,%llu\n", quat.w, quat.x, quat.y, quat.z, t];
    [self writeToLog:line];
}

-(void)logGyroscopeData:(CMRotationRate)gyro atTime:(unsigned long long)t
{
    if (!self.captureManager.isRecording) return;
    NSString *line = [NSString stringWithFormat:@"Gyroscope,%f,%f,%f,%llu\n", gyro.x, gyro.y, gyro.z, t];
    [self writeToLog:line];
}

// =======================
// General logging methods
// =======================
- (void)openLog {
    
    NSString *logName = [NSString stringWithFormat:@"%@.txt", self.fileName];
    self.logPath = [self.appDir stringByAppendingPathComponent:logName];
    
    //create file if it doesn't exist
    if(![[NSFileManager defaultManager] fileExistsAtPath:self.logPath])
        [[NSFileManager defaultManager] createFileAtPath:self.logPath contents:nil attributes:nil];
    
    //append text to file (you'll probably want to add a newline every write)
    self.logFile = [NSFileHandle fileHandleForUpdatingAtPath:self.logPath];
    [self.logFile seekToEndOfFile];
    
    logOpen = true;
}

- (void)writeToLog:(NSString*)line {
    if (!logOpen) return;
    [self.logFile writeData:[line dataUsingEncoding:NSUTF8StringEncoding]];
}

- (void)closeLog {
    logOpen = false;
    [self.logFile closeFile];
}



// =============================================================================
// Recording functions when the button is pressed

- (void)handleScan {
    // REC START
    if (!self.captureManager.isRecording) {
        
        // change UI
        [self.recBtn setImage:self.recStopImage
                     forState:UIControlStateNormal];
        self.lastBtn.enabled = false;
        self.nextBtn.enabled = false;
        
        NSURL* url = [self.captureManager startRecording];
        self.fileName = [[url lastPathComponent] stringByDeletingPathExtension];
        
    }
    // REC STOP
    else {
        
        isNeededToSave = YES;
        [self.captureManager stopRecording];
    }
}

- (void)handlePortrait {
    BOOL success = [self.captureManager captureImage];
    
    if (!success) { return; }

    // change UI
    [self.recBtn setAlpha:0.2];
    self.lastBtn.enabled = false;
    self.nextBtn.enabled = false;
}

- (void)handleQR {
    self.fiducialImg.hidden = false;

    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 500*NSEC_PER_MSEC), dispatch_get_main_queue(),
                   ^(void) {
        BOOL success = [self.captureManager captureImage];
        if (!success) { [self loadState]; }
    });
    
    // change UI
    [self.recBtn setAlpha:0.2];
    self.lastBtn.enabled = false;
    self.nextBtn.enabled = false;
}

- (void)handleIMU {
    // REC START
    if (!self.captureManager.isRecording) {
        
        // change UI
        [self.recBtn setImage:self.recStopImage
                     forState:UIControlStateNormal];
        self.lastBtn.enabled = false;
        self.nextBtn.enabled = false;
        
        NSURL* url = [self.captureManager startRecording];
        self.fileName = [[url lastPathComponent] stringByDeletingPathExtension];
        [self openLog];
        
    }
    // REC STOP
    else {
        
        isNeededToSave = YES;
        [self.captureManager stopRecording];
        [self closeLog];
        [self.sequences setIMULogForActiveSequence: [NSURL URLWithString:self.logPath ]];
    }
}

-(void)lastStep {
    self.lastBtn.enabled = false;
    self.nextBtn.enabled = false;
    [self.sequences lastStep];
    [self loadState];
}

-(void)nextStep {
    if (self.sequences.currentStep < 1) {
        self.lastBtn.enabled = false;
        self.nextBtn.enabled = false;
        [self.sequences nextStep];
        [self loadState];
    }
}


// =============================================================================
#pragma mark - AVCaptureManagerDeleagte

- (void)didFinishRecordingToOutputFileAtURL:(NSURL *)outputFileURL error:(NSError *)error {
    
    NSInteger state = self.sequences.currentStep;
    [self loadState];
    [self.recBtn setImage:self.recStartImage
                 forState:UIControlStateNormal];

    if (error) {
        NSLog(@"error:%@", error);
        return;
    }
    
    if (!isNeededToSave) {
        return;
    }
    
    if (state == 1) {
        [self.sequences setScaleVideoForActiveSequence:outputFileURL];
    }
    
    [self nextStep];
}

-(void)didFinishSavingImageAtURL:(NSURL *)outputFileURL error:(NSError *)error {
    // Recover the UI
    NSInteger state = self.sequences.currentStep;
    [self loadState];
    [self.recBtn setAlpha:1.0];
    
    if (error) {
        NSLog(@"Error Saving Image: %@", error);
        return;
    }
    
    if (state == 0) {
        [self.sequences setPortraitForActiveSequence:outputFileURL];
    }
    
    [self nextStep];
}


// =============================================================================
#pragma mark - IBAction

- (IBAction)recButtonTapped:(id)sender {
    NSInteger state = self.sequences.currentStep;
    
    if (state == 0) {
        // Take face portrait
        [self handlePortrait];
        
    } else if (state >= 1) {
//        // Record face scan
//        [self handleScan];
//        
//    } else if (state == 2) {
//        [self handleQR];
//        
//    } else if (state == 3) {
        [self handleIMU];

    }

}

- (IBAction)lastStepTapped:(id)sender {
    [self lastStep];
}

- (IBAction)nextStepTapped:(id)sender {
    [self nextStep];
}

@end
