import sys

import numpy as np
from skimage.transform import resize
import scipy
from PIL import ImageQt

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from Src.CelebA.preserving_gan import *

class DragRegionLabel(QLabel):
    def __init__(self, parent):
        super().__init__(parent)

        self.enb = False
        self.rub = QRubberBand(QRubberBand.Rectangle, self)
        self.pt = QPoint()

        self.x = -1
        self.y = -1
        self.w = 0
        self.h = 0

    def mousePressEvent(self, event):
        if self.enb==True and event.button()==Qt.LeftButton:
            self.pt = QPoint(event.pos())
            self.rub.setGeometry(QRect(self.pt, QSize()))
            self.rub.show()

    def mouseMoveEvent(self, event):
        if self.enb==True and not self.pt.isNull():
            self.rub.setGeometry(QRect(self.pt, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if self.enb==True and event.button()==Qt.LeftButton:
            # (x, y) -> (y, x)
            self.x = self.rub.pos().y()
            self.y = self.rub.pos().x()
            self.h = self.rub.size().height()
            self.w = self.rub.size().width()

            self.enb = False

    def reset(self):
        self.enb = False

        self.rub.hide()

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initResources()
        self.initUI()

    def initResources(self):
        self.pixfilenames = []
        self.pixmaps = []
        self.pixidx = 0

        self.selpixs = []
        self.regions = []

        self.drawpix = np.zeros((320, 320, 4), np.uint8)
        self.drawpix[:, :, 3] = 255

    def initUI(self):
        self.staBar = self.statusBar()
        self.staBar.showMessage('PreservingGAN Ready')

        self.impBtn = QPushButton('Import Image', self)
        self.impBtn.resize(120, 40)
        self.impBtn.move(10, 10)
        self.impBtn.clicked.connect(self.pressImpBtn)

        self.drawBtn = QPushButton('Draw', self)
        self.drawBtn.resize(80, 40)
        self.drawBtn.move(310, 10)
        self.drawBtn.clicked.connect(self.pressDrawBtn)

        self.prevBtn = QPushButton('<', self)
        self.prevBtn.resize(20, 20)
        self.prevBtn.move(10, 55)
        self.prevBtn.clicked.connect(self.pressPrevBtn)

        self.regBtn = QPushButton('O', self)
        self.regBtn.resize(20, 20)
        self.regBtn.move(40, 55)
        self.regBtn.clicked.connect(self.pressRegBtn)

        self.cclBtn = QPushButton('X', self)
        self.cclBtn.resize(20, 20)
        self.cclBtn.move(70, 55)
        self.cclBtn.clicked.connect(self.pressCclBtn)

        self.nextBtn = QPushButton('>', self)
        self.nextBtn.resize(20, 20)
        self.nextBtn.move(370, 55)
        self.nextBtn.clicked.connect(self.pressNextBtn)

        self.genBtn = QPushButton('Generate', self)
        self.genBtn.resize(100, 40)
        self.genBtn.move(680, 10)
        self.genBtn.clicked.connect(self.pressGenBtn)

        self.imgLbl = DragRegionLabel(self)
        self.imgLbl.resize(320, 320)
        self.imgLbl.move(40, 90)

        self.drawLbl = QLabel(self)
        self.drawLbl.resize(320, 320)
        self.drawLbl.move(440, 90)

        self.genpixs = []
        for i in range(9):
            self.genpixs.append(QLabel(self))
            self.genpixs[i].resize(120, 120)
            self.genpixs[i].move(800+(i//3)*120+(i//3)*10, 25+(i%3)*120+(i%3)*10)

        self.move(25, 25)
        self.setFixedSize(1200, 450)
        self.setWindowTitle('PreservingGAN - CelebA')

        self.show()
    
    def pressImpBtn(self):
        filename = QFileDialog.getOpenFileName()[0]
        if filename=='':
            return

        pix = QPixmap(filename)
        if pix.isNull()==True:
            return

        pix = pix.scaled(self.imgLbl.size())
        self.imgLbl.setPixmap(pix)

        self.pixfilenames.append(filename)
        self.pixmaps.append(pix)
        self.pixidx = len(self.pixmaps)-1

        self.imgLbl.reset()

    def pixmap2numpy(self, pix): # BGR->RGB
        w, h = 320, 320

        img = pix.toImage()
        s = img.bits().asstring(w*h*4)
        dat = np.fromstring(s, dtype=np.uint8).reshape((w, h, 4))
        dat = np.concatenate((dat[:, :, 2:3], dat[:, :, 1:2], dat[:, :, 0:1], dat[:, :, 3:4]), axis=2)

        return dat

    def numpy2pixmap(self, dat):
        pix = QPixmap.fromImage(ImageQt.ImageQt(scipy.misc.toimage(dat)))
        return pix

    def drawReg(self, dat, box):
        x, y, h, w = box

        for i in range(max(x, 0), min(x+h, 320)):
            for j in range(max(y, 0), min(y+w, 320)):
                self.drawpix[i, j, :] = dat[i, j, :]

    def pressDrawBtn(self):

        self.selpixs.append(self.pixmaps[self.pixidx])
        self.regions.append((self.imgLbl.x, self.imgLbl.y, self.imgLbl.h, self.imgLbl.w))

        dat = self.pixmap2numpy(self.pixmaps[self.pixidx])
        self.drawReg(dat, (self.imgLbl.x, self.imgLbl.y, self.imgLbl.h, self.imgLbl.w))
        pix = self.numpy2pixmap(self.drawpix)

        self.drawLbl.setPixmap(pix)

        self.imgLbl.reset()

    def pressPrevBtn(self):
        self.pixidx = (self.pixidx-1+len(self.pixmaps))%len(self.pixmaps)
        self.imgLbl.setPixmap(self.pixmaps[self.pixidx])

        self.imgLbl.reset()

    def pressNextBtn(self):
        self.pixidx = (self.pixidx+1+len(self.pixmaps))%len(self.pixmaps)
        self.imgLbl.setPixmap(self.pixmaps[self.pixidx])

        self.imgLbl.reset()

    def pressGenBtn(self):
        self.staBar.showMessage('Press Generation Button')

        img_cont = cv2.resize(self.drawpix, (64, 64))
        mask_pix = np.zeros((320, 320), np.float32)

        for reg in self.regions:
            x, y, h, w = reg

            for i in range(x, x+h):
                for j in range(y, y+w):
                    mask_pix[i][j] = 1

        mask_pix = cv2.resize(mask_pix, (64, 64))
        mask_lat = cv2.resize(mask_pix, (8, 8))

        out = produce(img_cont[:, :, :3], mask_pix, mask_lat)

        for i in range(9):
            dat = out[i]*255
            dat = dat.astype(np.uint8)
            tmp = np.ones((64, 64, 4), np.uint8)*255
            for x in range(64):
                for y in range(64):
                    tmp[x, y, :3] = dat[x][y]
            dat = tmp
            pix = self.numpy2pixmap(dat).scaled(120, 120)

            self.genpixs[i].setPixmap(pix)

    def pressRegBtn(self):
        self.imgLbl.reset()

        self.imgLbl.enb = True

    def pressCclBtn(self):
        self.imgLbl.reset()

if __name__=='__main__':
    app = QApplication(sys.argv)
    gui = GUI()

    sys.exit(app.exec_())
