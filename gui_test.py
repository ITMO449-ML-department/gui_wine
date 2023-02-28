import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImagePath
import math
import numpy as np
from numpy import array as npa
import os
import cv2
import rotate_and_cut.bb_getter2 as analyzer


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
    bb = analyzer.get_bb(app.filename, image_array=newImArray,
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
