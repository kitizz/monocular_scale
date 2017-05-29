//
//  SequenceTableViewController.swift
//  Camera_IMU
//
// Copyright Simon Lucey 2015, All rights Reserved......

import UIKit

class SequenceTableViewController: UITableViewController {

    @IBOutlet var sequenceTableView: UITableView!
    
    @IBOutlet weak var deleteBtn: UIBarButtonItem!
    @IBOutlet weak var editBtn: UIBarButtonItem!
//    let reuseIdentifier: String = "Cell"
//    var fileMgr: NSFileManager = NSFileManager.defaultManager()
//    var appDir: NSURL = NSURL()
    var sequences = Sequences()
    var selected: Int = -1;

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Uncomment the following line to preserve selection between presentations
        // self.clearsSelectionOnViewWillAppear = false

        // Uncomment the following line to display an Edit button in the navigation bar for this view controller.
        // self.navigationItem.rightBarButtonItem = self.editButtonItem()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        self.navigationController?.setNavigationBarHidden(false, animated: true)
        self.navigationController?.setToolbarHidden(false, animated: true)
        self.navigationItem.hidesBackButton = true

        sequences.readJsonIndex()
        tableView.reloadData()
        updateUI()
    }
    
    func updateUI() {
        if selected >= 0 {
            deleteBtn.isEnabled = true
            editBtn.isEnabled = true
        } else {
            deleteBtn.isEnabled = false
            editBtn.isEnabled = false
        }
    }
    
    @IBAction func deletePressed(_ sender: UIBarButtonItem) {
        if selected >= 0 {
            sequences.removeIndex(selected)
            tableView.reloadData()
            selected = -1
            updateUI()
        }
    }
    
    @IBAction func editPressed(_ sender: UIBarButtonItem) {
        if sequences.changeActiveSequence(selected) {
            self.performSegue(withIdentifier: "editSequence", sender: sender)
        }
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    // MARK: - Table view data source

    override func numberOfSections(in tableView: UITableView) -> Int {
        // #warning Potentially incomplete method implementation.
        // Return the number of sections.
        return 1
    }

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        // #warning Incomplete method implementation.
        // Return the number of rows in the section.
        return self.sequences.count()
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> SequenceCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "SequenceIdentifier", for: indexPath) as! SequenceCell

        // Configure the cell...
        cell.title.text = self.sequences.at(indexPath.row).name
        if let image = self.sequences.imageForSequence(indexPath.row) {
            cell.thumbnail.contentMode = .scaleAspectFill
            cell.thumbnail.clipsToBounds = true
            cell.thumbnail.image = image
        }

        return cell
    }
    
    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        selected = indexPath.row
        updateUI()
//        tableView.reloadRowsAtIndexPaths([indexPath], withRowAnimation: UITableViewRowAnimation.None)
    }

    /*
    // Override to support conditional editing of the table view.
    override func tableView(tableView: UITableView, canEditRowAtIndexPath indexPath: NSIndexPath) -> Bool {
        // Return NO if you do not want the specified item to be editable.
        return true
    }
    */

    /*
    // Override to support editing the table view.
    override func tableView(tableView: UITableView, commitEditingStyle editingStyle: UITableViewCellEditingStyle, forRowAtIndexPath indexPath: NSIndexPath) {
        if editingStyle == .Delete {
            // Delete the row from the data source
            tableView.deleteRowsAtIndexPaths([indexPath], withRowAnimation: .Fade)
        } else if editingStyle == .Insert {
            // Create a new instance of the appropriate class, insert it into the array, and add a new row to the table view
        }    
    }
    */

    /*
    // Override to support rearranging the table view.
    override func tableView(tableView: UITableView, moveRowAtIndexPath fromIndexPath: NSIndexPath, toIndexPath: NSIndexPath) {

    }
    */

    /*
    // Override to support conditional rearranging of the table view.
    override func tableView(tableView: UITableView, canMoveRowAtIndexPath indexPath: NSIndexPath) -> Bool {
        // Return NO if you do not want the item to be re-orderable.
        return true
    }
    */

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepareForSegue(segue: UIStoryboardSegue, sender: AnyObject?) {
        // Get the new view controller using [segue destinationViewController].
        // Pass the selected object to the new view controller.
    }
    */

}
