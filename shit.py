import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImagePath
import math
import numpy as np
from numpy import array as npa
import os
import cv2

import numpy as np
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.cluster as f
from sklearn.cluster import DBSCAN
from PIL import Image, ImageOps
import os
import scipy
from scipy.signal import find_peaks

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN


# old_version of _k_means_choose_channel
def _k_means_choose_channel_contours(new_X, image, verbose,save_path):
    first_channel = (new_X*(new_X == 1)).reshape((*image.shape[:-1], 1))
    second_channel = (new_X*(new_X == 2)).reshape((*image.shape[:-1], 1))
    third_channel = (new_X*(new_X == 3)).reshape((*image.shape[:-1], 1))
    if save_path is not None:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(first_channel,cmap="gray")
        ax[1].imshow(second_channel,cmap="gray")
        ax[2].imshow(third_channel,cmap="gray")
        fig.savefig(save_path+"kmeans.jpg")
        fig.clear()
    contours_counter = []
    for img in [first_channel, second_channel, third_channel]:
        img = np.array([img, img, img]).reshape((image.shape))
        img_grey = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        thresh = 0
        ret, thresh_img = cv2.threshold(img_grey.astype(
            np.uint8), thresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_counter.append(len(contours))
        if verbose == 2:
            print(f"   Amount {len(contours_counter)} -> {contours_counter[-1]}")

    goal_index = np.argmax(np.array(contours_counter)) + 1
    return goal_index

def _k_means_choose_channel(new_X, image, verbose,save_path):
    if save_path is not None:
        fig, ax = plt.subplots(1,3)
        print("   ", new_X.shape, image.shape)
        first_channel = (new_X*(new_X == 1)).reshape((*image.shape[:-1], 1))
        second_channel = (new_X*(new_X == 2)).reshape((*image.shape[:-1], 1))
        third_channel = (new_X*(new_X == 3)).reshape((*image.shape[:-1], 1))
        ax[0].imshow(first_channel,cmap="gray")
        ax[1].imshow(second_channel,cmap="gray")
        ax[2].imshow(third_channel,cmap="gray")
        fig.savefig(save_path+"kmeans.jpg")
        fig.clear()
    
    pixel_sum = [image.reshape((-1,3))[new_X==1].sum(),image.reshape((-1,3))[new_X==2].sum(),image.reshape((-1,3))[new_X==3].sum()]
    non_z = [np.count_nonzero(first_channel),np.count_nonzero(second_channel),np.count_nonzero(third_channel)]
    non_z[np.argmin(pixel_sum)] = max(non_z) + 1

    goal_index = 1 + np.argmin(non_z)
    return goal_index


def _get_kmeans_mask(image, verbose,save_path=None):
    predictions = KMeans(3,n_init="auto").fit_predict(image.reshape((-1, 3))) + 1
    goal_index = _k_means_choose_channel(predictions, image, verbose,save_path)
    lines_mask = (predictions*(predictions == goal_index)).reshape((*image.shape[:-1], 1))
    return lines_mask


def _get_hough(lines_mask, image):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 25  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    return cv2.HoughLinesP(lines_mask.astype("uint8"), rho, theta, threshold, np.array([]).astype("uint8"),
                            min_line_length, max_line_gap), line_image


def _get_hough_angles(lines_mask, image, save_path):
    lines, line_image = _get_hough(lines_mask, image)

    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angles.append(np.arctan2(y2-y1, x2-x1))
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    angles = np.array(angles)
    
    if save_path is not None:
        plt.imsave(save_path+"hough.jpg",line_image)
    return angles


def _get_rotation_angle_with_dbscan(angles):
    angles = (angles * 180 / np.pi).reshape((-1, 1))

    db = DBSCAN(eps=1, min_samples=100).fit(angles)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    card = []
    for i in range(n_clusters_):
        card.append((labels == i).sum())
    
    try:
        rot_angle = angles[labels == card.index(max(card))].mean()
    except ValueError:
        rot_angle = "error"
        print(card)
    print('   ', rot_angle)
    return rot_angle


def get_rotation_angle(name, image_array, verbose, save_path = None):
    """
    
    name : str
        path to image
    verbose : int {0, 1, 2}
        0 - no info; 1 - results, important info; 2 - every step

    return : int, np.array
        angle to rotate, mask made by k-means algo

    """
    if verbose >= 2:
        print(f"   ### Rotating {name}")

    if verbose == 2:
        print(f"   Openning image {name}")
    image = image_array if image_array is not None else cv2.imread(name)
    print('   ', image.shape)
    if verbose == 2:
        print("   Getting mask of rows using k-means algo")
    lines_mask = _get_kmeans_mask(image, verbose=verbose,save_path=save_path)

    if verbose == 2:
        print("   Calculating hough lines and angles")
    hough_angles = _get_hough_angles(lines_mask, image, save_path)

    return _get_rotation_angle_with_dbscan(hough_angles), lines_mask



def _rotate_bb(image_original, image_original_rotated_array, angle, bboxes, save_path):
    new_rect_rotated = np.zeros_like(image_original)
    colors = [(0,255,0), (255, 0, 0), (0,0,255)]
    i = 0
    rotated = []
    for bbox in bboxes:
        p1,p2,p3,p4 = bbox
        y1, x1 = p1
        y2, x2 = p2
        y3, x3 = p3
        y4, x4 = p4
        center = np.array([image_original_rotated_array.shape[1]//2,image_original_rotated_array.shape[0]//2]).astype(int)
        rotate_matrix = cv2.getRotationMatrix2D(center=center.tolist(), angle=angle, scale=1)
        rotate_matrix = rotate_matrix[:,:-1]
        p1 = np.array([x1 - center[0],y1 - center[1]]).dot(rotate_matrix).astype(int)
        p2 = np.array([x2 - center[0],y2 - center[1]]).dot(rotate_matrix).astype(int)
        p3 = np.array([x3 - center[0],y3 - center[1]]).dot(rotate_matrix).astype(int)
        p4 = np.array([x4 - center[0],y4 - center[1]]).dot(rotate_matrix).astype(int)
        center = np.array([new_rect_rotated.shape[1]//2,new_rect_rotated.shape[0]//2]).astype(int)
        new_rect_rotated = cv2.circle(new_rect_rotated, p1 + center,2, colors[i], 30)
        new_rect_rotated = cv2.circle(new_rect_rotated, p2 + center,2, colors[i], 30)
        new_rect_rotated = cv2.circle(new_rect_rotated, p3 + center,2, colors[i], 30)
        new_rect_rotated = cv2.circle(new_rect_rotated, p4 + center,2, colors[i], 30)
        rotated.append([p1+center, p2+center, p3+center, p4+center,])
        i = (i+1)%3
    final  = cv2.addWeighted(image_original,1,new_rect_rotated,1,0)
    if save_path is not None:
        plt.imsave(save_path + "bboxes.jpg", final)
    return rotated


def smooth(y, x):
    w = scipy.fftpack.rfft(y)
    f = scipy.fftpack.rfftfreq(len(x), x[1]-x[0])
    spectrum = w**2

    cutoff_idx = spectrum < (spectrum.max()/100)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    y2 = scipy.fftpack.irfft(w2)
    return y2

def _calculate_intensities_by_k_mean_mask(rows_image_rotated_array, image_original_rotated_array, verbose):
    if verbose == 2:
        print("   Calculating intensities")
    parts = rows_image_rotated_array.shape[0] // 6
    # parts = max(800, rows_image_rotated_array.shape[0] // 6)
    # print(parts)
    # parts = 800
    mn = []
    inds = np.linspace(0, rows_image_rotated_array.shape[0]-1, parts+1).astype(int)
    for i in range(1, parts+1):
        if np.count_nonzero(image_original_rotated_array[inds[i-1]:inds[i], :, 1]) > 0:
            mn.append(np.count_nonzero(rows_image_rotated_array[inds[i-1]:inds[i], :, 1])/np.count_nonzero(image_original_rotated_array[inds[i-1]:inds[i], :, 1]))
        else:
            mn.append(0)
    mn_smoothed = smooth(mn,[i for i in range(len(mn))])
    return mn, mn_smoothed, parts, inds

def _calculate_intensities_by_key_points(image_original_rotated_array, verbose):
    img = cv2.cvtColor(image_original_rotated_array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # find the keypoints with STAR
    kp = star.detect(img,None)
    if verbose > 1:
        print(f"   Found {len(kp)} keypoints")

    rows_image_rotated_array = image_original_rotated_array.copy() * 0
    rows_image_rotated_array=cv2.drawKeypoints(rows_image_rotated_array,kp,None)

    if verbose == 2:
        print("   Calculating intensities")
    parts = image_original_rotated_array.shape[0] // 6
    # parts = max(800, rows_image_rotated_array.shape[0] // 6)
    # print(parts)
    # parts = 800
    mn = []
    inds = np.linspace(0, image_original_rotated_array.shape[0]-1, parts+1).astype(int)
    for i in range(1, parts+1):
        if np.count_nonzero(image_original_rotated_array[inds[i-1]:inds[i], :, 1]) > 0:
            mn.append(np.count_nonzero(rows_image_rotated_array[inds[i-1]:inds[i], :, 1])/np.count_nonzero(image_original_rotated_array[inds[i-1]:inds[i], :, 1]))
        else:
            mn.append(0)
    mn_smoothed = smooth(mn,[i for i in range(len(mn))])
    # limit_smoothed = max(mn_smoothed)*0.4
    return mn, mn_smoothed, parts, inds


def _calculate_limit(mn, mn_smoothed, save_path, verbose):
    limit = max(sorted(mn)[:int(len(mn)*0.95)]) * 0.3
    limit_smoothed = max(sorted(mn_smoothed)[:int(len(mn_smoothed)*0.97)]) * 0.45
    if save_path is not None:
        if verbose == 2:
            print("   Plotting gists")
        plt.plot(mn, label='plot not')
        plt.plot([i for i in range(0, len(mn))], [limit] * len(mn), label='not')
        plt.plot([i for i in range(0, len(mn))], [limit_smoothed] * len(mn), label='smoothed')
        plt.plot(mn_smoothed, "-", label='plot smooth')
        plt.legend()
        plt.savefig(save_path + "rows_gists.jpg")
        plt.close()
    return limit, limit_smoothed
    

def _prepare_save_location(save_path, name, verbose):
    if save_path is not None:
        if verbose == 2:
            print("   Preparing save path")
        save_path = f"{save_path}/{name.split('/')[-1].split('.')[0]}/"
        if verbose > 0:
            print(f"   Saving to '{save_path}'")
        os.makedirs(save_path, exist_ok=True)
    return save_path


def _find_borders(parts, mn, inds, limit, verbose):
    if verbose == 2:
        print("   Searching for rows")
    borders = []
    low = 0
    high = 0
    previous_is_row = False
    for i in range(1, parts+1):
        if mn[i-1] > limit:
            if not previous_is_row:
                low = inds[i-1]
                high = inds[i]
                previous_is_row = True
            else:
                high  = inds[i]
        else:
            if previous_is_row:
                borders.append((low, high))
            previous_is_row = False
    if verbose > 0:
        print(f"   Found {len(borders)} rows")
    return borders


def _plot_orig_cut_kmeans(image_original_rotated_array, parts, mn, limit, inds, rows_image_rotated_array, save_path, verbose):
    if save_path is not None:
        if verbose == 2:
            print("   Plotting comparison")
        image_array_for_drawing = image_original_rotated_array.copy()
        for i in range(1, parts+1):
            if mn[i-1] < limit:
                image_array_for_drawing[inds[i-1]:inds[i], :, :] = np.zeros_like(image_array_for_drawing[inds[i-1]:inds[i], :, :])

        f, axs = plt.subplots(1, 3, figsize=(12,8))
        axs[0].imshow(image_original_rotated_array)
        axs[1].imshow(image_array_for_drawing)
        axs[2].imshow(rows_image_rotated_array)
        f.savefig(save_path + "rows_comparison.jpg")
        f.clear()

def _find_bboxes(borders, image_original_rotated_array, save_path, verbose):
    bboxes = []
    bboxes_debug = []
    if verbose == 2:
        print("   Calculating bboxes")
    for low, high in borders:
        strip = image_original_rotated_array[low:high, :, :]
        counter_left = 0
        counter_right = strip.shape[1]-1
        # print(strip.shape)
        while True:
            if np.any(strip[:, counter_left, :] != 0):
                break
            else:
                # print(strip[:, counter_left, :])
                counter_left += 1
                if counter_left == strip.shape[1]-1:
                    # print("limit")
                    break
        while True:
            if np.any(strip[:, counter_right, :] != 0):
                break
            else:
                counter_right-= 1
                if counter_right < 1:
                    # print("limit")
                    break
        bboxes_debug.append(((low, high, counter_left, counter_right)))
        bboxes.append(((low, counter_left), (low, counter_right), (high, counter_right), (high, counter_left)))
        
    if save_path is not None:
        image_array_for_drawing = image_original_rotated_array.copy() * 0 + 255
        # if verbose == 2:
        #     print("Plotting rows debug")
        # for low, high, left, right in bboxes_debug:
        #     image_array_for_drawing[low:high, left:right, :] = image_original_rotated_array[low:high, left:right, :]
        # # plt.imshow(image_array_for_drawing)
        # plt.imsave(save_path + "rows_debug.jpg", image_array_for_drawing)
        if verbose == 2:
            print("   Plotting rows check")
        colors = [(0,255,0), (255, 0, 0), (0,0,255)]
        i=0
        for box in bboxes:
            a, b = box[0]
            c, d = box[2]
            new_rect_rotated = cv2.rectangle(image_array_for_drawing.copy() * 0, (b, a), (d, c), colors[i], -1)
            i = (i+1)%3
            image_original_rotated_array  = cv2.addWeighted(image_original_rotated_array,1,new_rect_rotated,1,0)
        plt.imsave(save_path + "rows_check.jpg", image_original_rotated_array)

    if verbose > 0:    
        print(f"Found {len(bboxes)} bboxes!\n")
    return bboxes


def _analyse_peaks(mn,inds,save_path =None):
    peaks, _ = find_peaks(mn, distance=5)
    
    if save_path is not None:
        mn = np.array(mn)
        plt.plot(mn)
        plt.plot(peaks,mn[peaks],"x")
        plt.savefig(save_path+"peaks.jpg")
        plt.close()
    
    diff = np.diff(peaks).mean()
    borders=[]
    
    for i in range(len(peaks)):
        try:
            borders.append([inds[peaks[i]] - int(diff*0.4), inds[peaks[i] + int(diff*0.4)]])
        except:
            pass
    return borders
    
    


def get_bb(name, image_array = None, save_path=None, intensity = "keypoints", smooth = False, verbose = 0):
    """
    
    name : str
        path to image
    save_path : str
        save path for plots
    verbose : int
        0 - no info; 1 - results, important info; 2 - every step
    intensity : {'keypoint', 'kmeansmask'}
        way of calculating intensity of a row ('keypoint' - by keypoint; 'kmeansmask' - by k-means mask)
    smooth : bool
        smooth intensity hist

    return: arr[], float
        array of bounding boxes, angle of rotation

    """

    if verbose > 0:
        print(f"======== IMAGE {name} ========")

    if image_array is not None:
        image_array = image_array[:,:, :3]
    
    save_path = _prepare_save_location(save_path, name, verbose)

    angle, lines_mask = get_rotation_angle(name, image_array, verbose, save_path)

    if angle == "error":
        print("Not sure about angle")
        return

    if verbose == 2:
        print("   Preparing arrays")
    
    image_original = Image.fromarray(image_array) if image_array is not None else Image.open(name)
    image_original_array = np.array(image_original)
    image_original_rotated = image_original.rotate(angle, expand=True)
    image_original_rotated_array = np.array(image_original_rotated)

    lines_mask_tiled = np.zeros_like(image_original_array)
    lines_mask_tiled[:,:,0] = lines_mask[:,:,0]
    lines_mask_tiled[:,:,1] = lines_mask[:,:,0]
    lines_mask_tiled[:,:,2] = lines_mask[:,:,0]
    lines_mask_tiled *= 255

    rows_image_array = lines_mask_tiled
    rows_image_rotated = Image.fromarray(lines_mask_tiled).rotate(angle, expand=True)
    rows_image_rotated_array = np.array(rows_image_rotated)
    
    if save_path is not None:
        plt.imsave(save_path + "rows_mask.jpg", lines_mask_tiled)

    if intensity == "keypoints":
        mn, mn_smoothed, parts, inds = _calculate_intensities_by_key_points(image_original_rotated_array, verbose)
    elif intensity == "kmeansmask":
        mn, mn_smoothed, parts, inds = _calculate_intensities_by_k_mean_mask(rows_image_rotated_array, image_original_rotated_array, verbose)

    limit, limit_smoothed = _calculate_limit(mn, mn_smoothed, save_path, verbose)

    if smooth:
        mn = mn_smoothed
        limit = limit_smoothed

    _plot_orig_cut_kmeans(image_original_rotated_array, parts, mn, limit, inds, rows_image_rotated_array, save_path, verbose)
    
    borders = _find_borders(parts, mn, inds, limit, verbose)
    # borders = _analyse_peaks(mn,inds,save_path)

    bboxes = _find_bboxes(borders, image_original_rotated_array, save_path, verbose)

    return _rotate_bb(image_original_array, image_original_rotated_array, angle, bboxes, save_path)
                    


# TODO: all work here
def rotate_and_cut(app):
    img = app.pil_image
    area = app.area
    imArray = np.asarray(img)
    polygon = []
    for _, point in area:
        polygon.append(tuple(point[:-1].astype(int)))
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)
    mask = mask.reshape((*mask.shape, 1))
    newImArray = imArray.copy()
    newImArray *= mask

    counter_left = 0
    counter_down = 0
    counter_up = newImArray.shape[0]-1
    counter_right = newImArray.shape[1]-1
    while not np.any(newImArray[:, counter_left, :] != 0):
        counter_left += 1
        if counter_left == newImArray.shape[1]-1:
            break

    while not np.any(newImArray[:, counter_right, :] != 0):
        counter_right -= 1
        if counter_right < 1:
            break

    while not np.any(newImArray[counter_down, :, :] != 0):
        counter_down += 1
        if counter_down == newImArray.shape[0]-1:
            break

    while not np.any(newImArray[counter_up, :, :] != 0):
        counter_up -= 1
        if counter_up < 1:
            break

    newImArray = newImArray[counter_down:counter_up,
                            counter_left:counter_right, :]
    # newIm = Image.fromarray(newImArray, "RGB")  # may be we need to save it
    app.area.clear()
    app.draw_image(app.pil_image, app.area)
    bb = get_bb(app.filename, image_array=newImArray,
                         intensity="keypoints", smooth=False, save_path='debug', verbose=2)
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    i = 0
    blank = np.zeros_like(newImArray, dtype=np.uint8)

    try:
        for bbox in bb:
            p1, p2, p3, p4 = np.array(bbox, dtype=np.int32)
            i = (i+1) % 3
            cv2.fillPoly(blank, pts=[np.array([p1, p2, p3, p4])], color=colors[i])
    except TypeError:
        pass

    newImArray = cv2.addWeighted(newImArray, 1,  blank, 0.5, 0.0)
    img = np.array(img)
    img[counter_down:counter_up, counter_left:counter_right][newImArray > 0] = newImArray[newImArray > 0]
    app.pil_image = Image.fromarray(img)
    app.draw_image(app.pil_image, app.area)


def get_distance_to_segment(point, segment):
    """
    Функция вычисляет расстояние от точки до отрезка
    Args:
        point: Точка (x, y)
        segment: Отрезок, заданный двумя концами (x1, y1), (x2, y2)
    Returns: 
        Расстояние от точки до отрезка 
    """

    x, y = point
    x1, y1, x2, y2 = segment
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C*C + D*D
    param = -1.0
    if (len_sq != 0):
        param = dot / len_sq

    res = 0
    if (param < 0):
        xx = x1
        yy = y1
        res = -1
    elif (param > 1):
        xx = x2
        yy = y2
        res = 1
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return math.sqrt(dx*dx + dy*dy), res


class Application(tk.Frame):
    def __init__(self, master: tk.Tk = None):
        super().__init__(master)

        self.master.geometry("1280x720")

        self.filename = None
        self.pil_image = None
        self.my_title = "Image Viewer"
        self.theme_color = "white"

        # ウィンドウの設定
        self.master.title(self.my_title)

        self.create_menu()
        self.create_widget()
        self.reset_transform()

        self.area = []
        self.chosen_point = None
        self.closest_point = None
        self.closest_distance = 10e300
        self.prev_closest_point = None
        self.prev_closest_distance = 10e300

        empty_im = Image.new("RGBA", (30, 30))
        ImageDraw.Draw(empty_im).ellipse(
            [(0, 0), (20, 20)], fill=(255, 0, 0, 127))
        self.empty_im = ImageTk.PhotoImage(empty_im)

        circled_img = Image.new("RGBA", (30, 30))
        ImageDraw.Draw(circled_img).ellipse([(0, 0), (20, 20)], fill=(
            255, 0, 0, 200), outline=(100, 0, 0, 255), width=3)
        self.circled_img = ImageTk.PhotoImage(circled_img)

        self.polygon_img = None

        self.i = 0

        self.master.bind('<Return>', lambda event: rotate_and_cut(self))

    def menu_open_clicked(self, event=None):
        filename = tk.filedialog.askopenfilename(filetypes=[("Image file", ".bmp .png .jpg .JPG .tif .TIF .tiff .TIFF"),
                                                            ("Bitmap", ".bmp"),
                                                            ("PNG", ".png"),
                                                            ("JPEG", ".jpg .JPG"),
                                                            ("Tiff", ".tif .TIF .tiff .TIFF")],
                                                 initialdir=os.getcwd())
        self.set_image(filename)

    def menu_quit_clicked(self):
        self.master.destroy()

    def create_menu(self):
        self.menu_bar = tk.Menu(self)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=tk.OFF)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(
            label="Open", command=self.menu_open_clicked, accelerator="Ctrl+O")
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Exit", command=self.menu_quit_clicked)
        self.menu_bar.bind_all("<Control-o>", self.menu_open_clicked)

        self.view_menu = tk.Menu(self.menu_bar, tearoff=tk.OFF)
        self.menu_bar.add_cascade(label="View", menu=self.view_menu)
        self.theme_menu = tk.Menu(self.view_menu, tearoff=tk.OFF)
        self.view_menu.add_cascade(label="Theme", menu=self.theme_menu)
        self.theme_menu.add_command(
            label="Dark",  command=lambda: self.set_theme("black"))
        self.theme_menu.add_command(
            label="Light", command=lambda: self.set_theme("white"))

        self.master.config(menu=self.menu_bar)

    def create_widget(self):

        frame_statusbar = tk.Frame(self.master, bd=1, relief=tk.SUNKEN)
        self.label_image_info = tk.Label(
            frame_statusbar, text="image info", anchor=tk.E, padx=5)
        self.label_image_pixel = tk.Label(
            frame_statusbar, text="(x, y)", anchor=tk.W, padx=5)
        self.label_image_info.pack(side=tk.RIGHT)
        self.label_image_pixel.pack(side=tk.LEFT)
        frame_statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(self.master, background="white")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        self.master.bind("<Button-1>", self.mouse_down_left)
        self.master.bind("<ButtonRelease-1>", self.mouse_up)
        self.master.bind("<B1-Motion>", self.mouse_move_left)
        self.master.bind("<Motion>", self.mouse_move)
        self.master.bind("<Double-Button-1>", self.put_point)
        self.master.bind("<Button-2>", self.flash_zoom)
        self.master.bind("<MouseWheel>", self.mouse_wheel)
        self.master.bind("<Delete>", self.delete_point)
        self.master.bind("<BackSpace>", self.delete_point)

    def set_theme(self, color):
        self.theme_color = color
        self.canvas.configure(background=color)

    def set_image(self, filename):
        if not filename:
            return
        self.pil_image = Image.open(filename)
        self.filename = filename
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.draw_image(self.pil_image, self.area)

        self.master.title(self.my_title + " - " + os.path.basename(filename))
        self.label_image_info["text"] = f"{self.pil_image.format} : {self.pil_image.width} x {self.pil_image.height} {self.pil_image.mode}"
        os.chdir(os.path.dirname(filename))

    def mouse_up(self, event=None):
        self.chosen_point = None

    def mouse_down_left(self, event):
        if self.closest_distance < 20**2:
            self.chosen_point = self.closest_point
        else:
            self.chosen_point = None

        self.__old_event = event

    def mouse_move_left(self, event):
        if self.pil_image == None:
            return
        if self.chosen_point is not None:
            mouse_point_coords = self.to_image_point(event.x, event.y)
            self.chosen_point[1] = mouse_point_coords
        else:
            self.translate(event.x - self.__old_event.x,
                           event.y - self.__old_event.y)
        self.redraw_image()
        self.__old_event = event

    def mouse_move(self, event):
        if (self.pil_image is None):
            return

        closest_point = None
        closest_distance = 1e100
        for point in self.area:
            point[0] = self.empty_im
            canvas_coords = self.mat_affine @ point[1]
            current_distance = (
                canvas_coords[0] - event.x) ** 2 + (canvas_coords[1] - event.y) ** 2
            if current_distance < closest_distance:
                closest_distance = current_distance
                closest_point = point
        self.set_closest_point(closest_point, closest_distance)

        if self.closest_distance < 20 ** 2:
            self.master.configure(cursor="fleur")
            self.closest_point[0] = self.circled_img
        else:
            self.master.configure(cursor="arrow")

        if (self.closest_distance - 20**2) * (self.prev_closest_distance - 20**2) <= 0:
            self.redraw_image()
        elif (self.closest_distance - 20**2) <= 0 and (self.prev_closest_distance - 20**2) <= 0 and \
                self.prev_closest_point != self.closest_point:
            self.redraw_image()

        image_point = self.to_image_point(event.x, event.y)
        self.label_image_pixel["text"] = (
            f"({image_point[0]:.2f}, {image_point[1]:.2f})")
        self.label_image_pixel.configure(fg="black")
        if image_point[0] < 0 or image_point[1] < 0 or \
           image_point[0] > self.pil_image.width or image_point[1] > self.pil_image.height:
            self.label_image_pixel.configure(fg="red")
        else:
            self.label_image_pixel["text"] = ("(--, --)")

    def flash_zoom(self, event):
        if self.pil_image is None:
            return
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.redraw_image()

    def mouse_wheel(self, event):
        if self.pil_image == None:
            return

        if event.state == 8:
            if (event.delta < 0):
                self.scale_at(0.8, event.x, event.y)
            else:
                self.scale_at(1.25, event.x, event.y)
        self.redraw_image()

    def reset_transform(self):
        self.mat_affine = np.eye(3)

    def translate(self, offset_x, offset_y):
        mat = np.eye(3)
        mat[0, 2] = float(offset_x)
        mat[1, 2] = float(offset_y)
        self.mat_affine = mat @ self.mat_affine

    def scale(self, scale: float):
        self.mat_affine = np.diag([scale, scale, 1]) @ self.mat_affine

    def scale_at(self, scale: float, cx: float, cy: float):

        self.translate(-cx, -cy)
        self.scale(scale)
        self.translate(cx, cy)

    def zoom_fit(self, image_width, image_height):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if (image_width * image_height <= 0) or (canvas_width * canvas_height <= 0):
            return

        self.reset_transform()

        scale = 1.0
        offsetx = 0.0
        offsety = 0.0

        if (canvas_width * image_height) > (image_width * canvas_height):
            scale = canvas_height / image_height
            offsetx = (canvas_width - image_width * scale) / 2
        else:
            scale = canvas_width / image_width
            offsety = (canvas_height - image_height * scale) / 2

        self.scale(scale)
        self.translate(offsetx, offsety)

    def to_image_point(self, x, y):
        if self.pil_image is None:
            return []
        image_point = np.linalg.inv(self.mat_affine) @ np.array([x, y, 1.])

        return image_point

    def draw_image(self, pil_image, points):
        if pil_image is None:
            return

        self.pil_image = pil_image

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        mat_inv = np.linalg.inv(self.mat_affine)

        affine_inv = mat_inv[:-1, :].flatten()

        dst = self.pil_image.transform((canvas_width, canvas_height), Image.AFFINE, affine_inv,
                                       Image.NEAREST, fillcolor=(255, 255, 255, 0))

        self.image = ImageTk.PhotoImage(image=dst)
        self.canvas.delete("all")
        item = self.canvas.create_image(0, 0, anchor='nw', image=self.image)

        canvas_coords = []
        for point in points:
            canvas_coords.append(tuple((self.mat_affine @ point[1])[0:2]))

        if len(canvas_coords) > 2:
            polygon_img = Image.new(
                "RGBA", (self.image.width(), self.image.height()))
            ImageDraw.Draw(polygon_img).polygon(
                canvas_coords, fill=(255, 0, 0, 100), outline=(255, 0, 0, 100))

            self.polygon_img = ImageTk.PhotoImage(polygon_img)
            item = self.canvas.create_image(
                0, 0, image=self.polygon_img, anchor='nw')

    def redraw_image(self):
        if self.pil_image is None:
            return
        self.draw_image(self.pil_image, self.area)

    def put_point(self, event):
        if self.pil_image is None:
            return

        min_dist = 1e100
        ind = len(self.area)
        insert = ind
        closest = 0
        point_coords = self.to_image_point(event.x, event.y)
        added_point = [self.circled_img, point_coords]
        if len(self.area) > 2:
            for i, point in enumerate(self.area):
                point[0] = self.empty_im
                first = point[1][0:2]
                second = self.area[(i+1) % len(self.area)][1][0:2]

                curr_dist, curr_closest = get_distance_to_segment(
                    point_coords[0:2], [*first, *second])

                if curr_dist < min_dist:
                    min_dist = curr_dist
                    closest = curr_closest
                    ind = i+1
                    insert = ind

            if closest != 0:
                if closest == -1:
                    ind_x1 = (ind-2) % len(self.area)
                    ind_x2 = (ind-1) % len(self.area)
                    ind_x3 = (ind-0) % len(self.area)
                elif closest == 1:
                    ind_x1 = (ind - 1) % len(self.area)
                    ind_x2 = (ind + 0) % len(self.area)
                    ind_x3 = (ind + 1) % len(self.area)

                x2x1 = (npa(self.area[ind_x1][1]) -
                        npa(self.area[ind_x2][1]))[0:2]
                x2x3 = (npa(self.area[ind_x3][1]) -
                        npa(self.area[ind_x2][1]))[0:2]
                x2xp = (npa(point_coords) - npa(self.area[ind_x2][1]))[0:2]
                x2x1 /= np.linalg.norm(x2x1)
                x2x3 /= np.linalg.norm(x2x3)
                x2xp /= np.linalg.norm(x2xp)

                bisectr = -(x2x1 + x2x3)/2

                # res1 = np.linalg.inv(np.c_[x2x1.reshape((-1, 1)), bisectr.reshape((-1, 1))]) @ x2xp
                # res2 = np.linalg.inv(np.c_[x2x3.reshape((-1, 1)), bisectr.reshape((-1, 1))]) @ x2xp
                if np.all(np.linalg.inv(np.c_[x2x1.reshape((-1, 1)), bisectr.reshape((-1, 1))]) @ x2xp > 0):
                    insert = ind_x2
                else:
                    insert = ind_x3

        self.area.insert(insert, added_point)

        self.set_closest_point(added_point, 0)
        self.master.configure(cursor="fleur")

        self.redraw_image()

    def delete_point(self, event):
        if self.closest_distance <= 20**2:
            self.area.remove(self.closest_point)
        closest_point = None
        closest_distance = 1e100
        for point in self.area:
            point[0] = self.empty_im
            canvas_coords = self.mat_affine @ point[1]
            current_distance = (
                canvas_coords[0] - event.x) ** 2 + (canvas_coords[1] - event.y) ** 2
            if current_distance < closest_distance:
                closest_distance = current_distance
                closest_point = point
        self.set_closest_point(closest_point, closest_distance)
        if self.closest_distance < 20 ** 2:
            self.master.configure(cursor="fleur")
            self.closest_point[0] = self.circled_img
        else:
            self.master.configure(cursor="arrow")
        self.redraw_image()
        # if (self.closest_distance - 20**2) * (self.prev_closest_distance - 20**2) <= 0:
        # self.redraw_image()
        # elif (self.closest_distance - 20**2) <= 0 and (self.prev_closest_distance - 20**2) <= 0 and \
        #     self.prev_closest_point != self.closest_point:
        # self.redraw_image()

    def set_closest_point(self, point, distance):
        self.prev_closest_point = self.closest_point
        self.prev_closest_distance = self.closest_distance
        self.closest_point = point
        self.closest_distance = distance


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
