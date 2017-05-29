//
//  AddSequenceViewController.swift
//  Camera_IMU
//
// Copyright Simon Lucey 2015, All rights Reserved......

import UIKit

class AddSequenceViewController: UIViewController, UINavigationControllerDelegate, UITextFieldDelegate {

    @IBOutlet weak var sequenceName: UITextField!
    @IBOutlet weak var nextBtn: UIBarButtonItem!
    @IBOutlet weak var errorLabel: UILabel!
    
    override func viewDidLoad() {
        sequenceName.delegate = self
    }
    
    @IBAction func sequenceNameChanged(_ sender: AnyObject) {
        // Check if the input name is valid, and enable to the button if so
        nextBtn.isEnabled = !sequenceName.text!.isEmpty
        errorLabel.text = ""
    }
    
    @IBAction func nextNavigation(_ sender: AnyObject) {
        next()
    }

    func next() -> Bool {
        let seq = Sequences()
        let success = seq.beginRecording(sequenceName.text!)

        if success {
            self.performSegue(withIdentifier: "nextSegue", sender: self)
            return true
            
        } else {
            errorLabel.text = "Sequence name already exists"
            return false
        }
    }
    
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        if next() {
            textField.resignFirstResponder()
            return true
            
        } else {
            return false
        }
    }
}
