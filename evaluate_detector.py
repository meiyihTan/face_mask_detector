import os
import glob
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import json
import operator
import math
import argparse
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt 
from data import AnnotationTransform, FaceMaskData, detection_collate, cfg
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.pred_utils import *
from evaluation.eval_utils import * 

parser = argparse.ArgumentParser(description='FaceBoxes evaluation')
parser.add_argument('--dataset', default='data/FaceMask', help='Dataset directory')
parser.add_argument('--model_checkpoints', default='model_checkpoints/checkpoint_FaceBoxes.pth', help='Trained model checkpoint directory')
parser.add_argument('--confidence_threshold', default=0.15, type=float, help='confidence threshold')
parser.add_argument('--nms_threshold', default=0.15, type=float, help='threshold threshold')
parser.add_argument('--show_animation', default=True, help='show animation of every detections')
parser.add_argument('--draw_plot', default=True, help='plot eval graphs')
parser.add_argument('--eval_folder', default='evaluation/', help='path to store input files for evaluation')

args = parser.parse_args()

dataset = args.dataset
val_datapath = os.path.join(dataset, 'val')
model_checkpoints = torch.load(args.model_checkpoints)
show_animation = args.show_animation 
confidence_threshold = args.confidence_threshold
nms_threshold = args.nms_threshold
eval_folder = args.eval_folder
input_filepath = os.path.join(eval_folder, 'input') 
output_filepath = os.path.join(eval_folder, 'output') 
draw_plot = args.draw_plot

# create directories 
if os.path.join(eval_folder)==False: 
    os.makedirs(eval_folder)
    os.makedirs(input_filepath)
    os.makedirs(output_filepath)
    
MINOVERLAP = 0.5

# generate files needed for evaluation 
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

def is_image(path): 
    if path.split('.')[-1] in ['png', 'jpg', 'jpeg']:
        return True 
    return False

classes = {'background': 0, 'face': 1, 'fask_mask': 2}

model = FaceBoxes(phase='test', size=None, num_classes=len(classes), pretrained=False)
model.load_state_dict(model_checkpoints)
model.eval() 
model = model.to(device)
print('Finished loading model...')

val_dataset = FaceMaskData(root=dataset, split='val', preproc=None, target_transform=AnnotationTransform())
num_images = len(val_dataset) # number of validation images 

gt_path = os.path.join(input_filepath, 'ground_truths')
dr_path = os.path.join(input_filepath, 'detection_results')
# create .txt file for ground-truth data
if not os.path.exists(gt_path): # if it doesn't exist already
    os.makedirs(gt_path)

if not os.path.exists(dr_path): # if it doesn't exist already
    os.makedirs(dr_path)
    
create_gt_file(model, gt_path, val_dataset)
create_result_file(model, dr_path, val_dataset, confidence_threshold, nms_threshold)

IMG_PATH = os.path.join(os.getcwd(), dataset)
if os.path.exists(IMG_PATH): 
    for dirpath, dirnames, files in os.walk(IMG_PATH):
        if not files: 
            show_animation=False
else: 
    show_animation=True


TEMP_FILES_PATH = os.path.join(eval_folder, 'temp')
if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
    os.makedirs(TEMP_FILES_PATH)
output_files_path = os.path.join(eval_folder, 'output')
if os.path.exists(output_files_path): # if it exist already
    # reset the output directory
    shutil.rmtree(output_files_path)
    
os.makedirs(output_files_path)

if show_animation:
    os.makedirs(os.path.join(output_files_path, "images", "detections_one_by_one"))
    
IMG_PATH = os.path.join(os.getcwd(), val_datapath)
if os.path.exists(IMG_PATH): 
    for dirpath, dirnames, files in os.walk(IMG_PATH):
        if not files: 
            show_animation=False
else: 
    show_animation=True

    
ground_truth_files_list = [os.path.join(gt_path, i) for i in os.listdir(gt_path)]
if len(ground_truth_files_list)==0: 
    raise Exception("Error: No ground-truth files found!")

gt_counter_per_class = {}
counter_images_per_class = {}
    
ground_truth_files_list.sort()
gt_files = [] 
for txt_file in ground_truth_files_list: 
    file_id = os.path.basename(txt_file).split(".txt",1)[0]
    temp_path = os.path.join(dr_path, (file_id + ".txt"))
    if not os.path.exists(temp_path):
        error_msg = "Error. File not found: {}\n".format(temp_path)
        raise Exception(error_msg)
    
    lines_list = file_lines_to_list(txt_file) # read gt files 
    bounding_boxes = []
    already_seen_classes = []
    
    for line in lines_list: 
        class_name, left, top, right, bottom = line.split()
        bbox = left + " " + top + " " + right + " " +bottom
        bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
        # count that object

        if class_name in gt_counter_per_class:
            gt_counter_per_class[class_name] += 1
        else:
            # if class didn't exist yet
            gt_counter_per_class[class_name] = 1

        if class_name not in already_seen_classes:
            if class_name in counter_images_per_class:
                counter_images_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                counter_images_per_class[class_name] = 1
            already_seen_classes.append(class_name)    
            
    new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
    gt_files.append(new_temp_file)
    with open(new_temp_file, 'w') as outfile:
        json.dump(bounding_boxes, outfile)
        
gt_classes = list(gt_counter_per_class.keys())
# let's sort the classes alphabetically
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)

dr_files_list = [os.path.join(dr_path, i) for i in os.listdir(dr_path)]
dr_files_list.sort()

for class_index, class_name in enumerate(gt_classes):
    bounding_boxes = []
    for txt_file in dr_files_list:
        file_id = os.path.basename(txt_file).split(".txt",1)[0]
        temp_path = os.path.join(gt_path, (file_id + ".txt"))
        
        if class_index == 0:
            if not os.path.exists(temp_path):
                error_msg = "Error. File not found: {}\n".format(temp_path)
                raise Exception(error_msg)
                
        lines = file_lines_to_list(txt_file)
        
        for line in lines:
            try:
                tmp_class_name, confidence, left, top, right, bottom = line.split()
                
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                raise Exception(error_msg)
                
            if tmp_class_name == class_name:
                bbox = left + " " + top + " " + right + " " +bottom
                bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
                #print(bounding_boxes)
    # sort detection-results by decreasing confidence
    bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
    with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)
        
# calculate AP for each class 
sum_AP = 0.0
ap_dictionary = {}
lamr_dictionary = {}
# open file to store the output
with open(output_files_path + "/output.txt", 'w') as output_file:
    output_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        """
         Load detection-results of that class
        """
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))

        """
         Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            
            if show_animation:
                # find ground truth image
                ground_truth_img = [] 
                patterns = (".jpg", ".png")
                for p in patterns: 
                    ground_truth_img.extend(glob.glob1(IMG_PATH, file_id + p))
                
                #tifCounter = len(glob.glob1(myPath,"*.tif"))
                if len(ground_truth_img) == 0:
                    raise Exception("Error. Image not found with id: " + file_id)
                elif len(ground_truth_img) > 1:
                    raise Exception("Error. Multiple image with id: " + file_id)
                else: # found image
                    #print(IMG_PATH + "/" + ground_truth_img[0])
                    # Load image
                    img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                    # load image with draws of multiple detections
                    img_cumulative_path = output_files_path + "/images/" + ground_truth_img[0]
                    if os.path.isfile(img_cumulative_path):
                        img_cumulative = cv2.imread(img_cumulative_path)
                    else:
                        img_cumulative = img.copy()
                    # Add bottom border to image
                    bottom_border = 60
                    BLACK = [0, 0, 0]
                    img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)

            
            # assign detection-results to ground truth object if any
            # open ground-truth with that file_id
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [ float(x) for x in detection["bbox"].split() ]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [ float(x) for x in obj["bbox"].split() ]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
                            
                            
            if show_animation:
                status = "NO MATCH FOUND!" # status is only used in the animation
     
            # set minimum overlap
            min_overlap = MINOVERLAP
            
            if ovmax >= min_overlap:
                if not bool(gt_match["used"]):
                    # true positive
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[class_name] += 1
                    # update the ".json" file
                    with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    if show_animation:
                        status = "MATCH!"
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
                    if show_animation:
                        status = "REPEATED MATCH!"

            """
             Draw image to show animation
            """
            if show_animation:
                height, widht = img.shape[:2]
                # colors (OpenCV works with BGR)
                white = (255,255,255)
                light_blue = (255,200,100)
                green = (0,255,0)
                light_red = (30,30,255)
                # 1st line
                margin = 10
                v_pos = int(height - margin - (bottom_border / 2.0))
                text = "Image: " + ground_truth_img[0] + " "
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                if ovmax != -1:
                    color = light_red
                    if status == "INSUFFICIENT OVERLAP":
                        text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
                    else:
                        text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
                        color = green
                    img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                # 2nd line
                v_pos += int(bottom_border / 2.0)
                rank_pos = str(idx+1) # rank position (idx starts at 0)
                text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(detection["confidence"])*100)
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                color = light_red
                if status == "MATCH!":
                    color = green
                text = "Result: " + status + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if ovmax > 0: # if there is intersections between the bounding-boxes
                    bbgt = [ int(round(float(x))) for x in gt_match["bbox"].split() ]
                    cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                    cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                    cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
                bb = [int(i) for i in bb]
                cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                # show image
                cv2.imshow("Animation", img)
                cv2.waitKey(20) # show for 20 ms
                # save image to output
                output_img_path = output_files_path + "/images/detections_one_by_one/" + class_name + "_detection" + str(idx) + ".jpg"
                cv2.imwrite(output_img_path, img)
                # save the image with all the objects drawn to it
                cv2.imwrite(img_cumulative_path, img_cumulative)

        #print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            deno = (fp[idx] + tp[idx])
            # avoid zero-division error
            if deno == 0:
                prec[idx] = 0 
            if deno > 0: 
                prec[idx] = float(tp[idx]) / deno
        #print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
        """
         Write to output.txt
        """
        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]
        output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        print(text)
        ap_dictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
        lamr_dictionary[class_name] = lamr

        """
         Draw plot
        """
        if draw_plot:
            plt.plot(rec, prec, '-o')
            # add a new penultimate point to the list (mrec[-2], 0.0)
            # since the last line segment (and respective area) do not affect the AP value
            area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
            area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
            plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
            # set window title
            fig = plt.gcf() # gcf - get current figure
            fig.canvas.set_window_title('AP ' + class_name)
            # set plot title
            plt.title('class: ' + text)
            #plt.suptitle('This is a somewhat long figure title', fontsize=16)
            # set axis titles
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # optional - set axes
            axes = plt.gca() # gca - get current axes
            axes.set_xlim([0.0,1.0])
            axes.set_ylim([0.0,1.05]) # .05 to give some extra space
            # Alternative option -> wait for button to be pressed
            #while not plt.waitforbuttonpress(): pass # wait for key display
            # Alternative option -> normal display
            #plt.show()
            # save the plot
            classes_eval_path = output_files_path + "/classes/"
            if os.path.exists(classes_eval_path)==False:
                os.makedirs(classes_eval_path)
            fig.savefig(classes_eval_path + class_name + ".png")
            plt.cla() # clear axes for next plot

    if show_animation:
        cv2.destroyAllWindows()

    output_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    output_file.write(text + "\n")
    print(text)

"""
 Draw false negatives
"""
if show_animation:
    pink = (203,192,255)
    for tmp_file in gt_files:
        ground_truth_data = json.load(open(tmp_file))
        #print(ground_truth_data)
        # get name of corresponding image
        start = TEMP_FILES_PATH + '/'
        img_id = tmp_file[tmp_file.find(start)+len(start):tmp_file.rfind('_ground_truth.json')]
        img_cumulative_path = output_files_path + "/images/" + img_id + ".jpg"
        img = cv2.imread(img_cumulative_path)
        if img is None:
            img_path = IMG_PATH + '/' + img_id + ".jpg"
            if os.path.exists(img_path)==False: 
                img_path = IMG_PATH + '/' + img_id + ".png"
            img = cv2.imread(img_path)
        # draw false negatives
        for obj in ground_truth_data:
            if not obj['used']:
                bbgt = [ int(round(float(x))) for x in obj["bbox"].split() ]
                cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),pink,2)
        cv2.imwrite(img_cumulative_path, img)

# remove the temp_files directory
shutil.rmtree(TEMP_FILES_PATH)

"""
 Count total of detection-results
"""
# iterate through all the files
det_counter_per_class = {}
for txt_file in dr_files_list:
    # get lines to list
    lines_list = file_lines_to_list(txt_file)
    for line in lines_list:
        class_name = line.split()[0]
        # count that object
        if class_name in det_counter_per_class:
            det_counter_per_class[class_name] += 1
        else:
            # if class didn't exist yet
            det_counter_per_class[class_name] = 1
#print(det_counter_per_class)
dr_classes = list(det_counter_per_class.keys())


"""
 Plot the total number of occurences of each class in the ground-truth
"""
if draw_plot:
    window_title = "ground-truth-info"
    plot_title = "ground-truth\n"
    plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = output_files_path + "/ground-truth-info.png"
    to_show = False
    plot_color = 'forestgreen'
    draw_plot_func(
        gt_counter_per_class,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
        )

"""
 Write number of ground-truth objects per class to results.txt
"""
with open(output_files_path + "/output.txt", 'a') as output_file:
    output_file.write("\n# Number of ground-truth objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        output_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

"""
 Finish counting true positives
"""
for class_name in dr_classes:
    # if class exists in detection-result but not in ground-truth then there are no true positives in that class
    if class_name not in gt_classes:
        count_true_positives[class_name] = 0
#print(count_true_positives)

"""
 Plot the total number of occurences of each class in the "detection-results" folder
"""
if draw_plot:
    window_title = "detection-results-info"
    # Plot title
    plot_title = "detection-results\n"
    plot_title += "(" + str(len(dr_files_list)) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = output_files_path + "/detection-results-info.png"
    to_show = False
    plot_color = 'forestgreen'
    true_p_bar = count_true_positives
    draw_plot_func(
        det_counter_per_class,
        len(det_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        true_p_bar
        )

"""
 Write number of detected objects per class to output.txt
"""
with open(output_files_path + "/output.txt", 'a') as output_file:
    output_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(dr_classes):
        n_det = det_counter_per_class[class_name]
        text = class_name + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
        output_file.write(text)

"""
 Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
"""
if draw_plot:
    window_title = "lamr"
    plot_title = "log-average miss rate"
    x_label = "log-average miss rate"
    output_path = output_files_path + "/lamr.png"
    to_show = False
    plot_color = 'royalblue'
    draw_plot_func(
        lamr_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
        )

"""
 Draw mAP plot (Show AP's of all classes in decreasing order)
"""
if draw_plot:
    window_title = "mAP"
    plot_title = "mAP = {0:.2f}%".format(mAP*100)
    x_label = "Average Precision"
    output_path = output_files_path + "/mAP.png"
    to_show = True
    plot_color = 'royalblue'
    draw_plot_func(
        ap_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
        )
