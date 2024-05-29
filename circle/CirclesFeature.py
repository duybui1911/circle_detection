import cv2
import numpy as np
from scipy import optimize
from math import fabs
from circle.MathFunc.Point2 import Point2
from circle.MathFunc.Constants import RANSAC_parameter


class ExtractedCircles():
    _centre = Point2(0, 0)
    _radius = 0

    def __init__(self):

        self._centre = Point2(0, 0)
        self._radius = 0
        self._pixels = []

    def __eq__(self, other):
        tol = 4.0
        return (
            fabs(self._centre - other._centre) < tol
            and fabs(self._radius - other._radius) < tol
        )

    def __hash__(self):
        return hash((self._centre, self._radius))

    def __repr__(self):
        return "".join([
            "Circle (Centre =", str(self._centre),
            ", Radius =", str(self._radius), ")"
        ])

    @staticmethod
    def ExtractCircle(self, centre, radius):
        self._centre = centre
        self._radius = radius


class CirclesFeature():

    global threshImg, ImgHeight, ImgWidth, ImgChannels

    @staticmethod
    def Detect(img, debug=False):
        # def Detect(Feature_Manager, debug=False):
        global threshImg
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshImg = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite('thresh.jpg', threshImg)
        contours = CirclesFeature.preprocess(img)
        DetectedCircles = CirclesFeature.fitting(img, contours)
        DetectedCircles = CirclesFeature.UniqueCircles(DetectedCircles)
        RequiredCircles = CirclesFeature.circleFiltering(DetectedCircles)

        return RequiredCircles

    @staticmethod
    def preprocess(img):
        global ImgHeight, ImgWidth, ImgChannels
        ImgHeight, ImgWidth, ImgChannels = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_thresh = cv2.threshold(
            img_gray, np.mean(img_gray), 255, cv2.THRESH_BINARY_INV)# | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(
            img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # RETR_CCOMP
        cv2.imwrite('thresh2.jpg', img_thresh)
        return contours

    @staticmethod
    def fitting(img, contours):
        global ImgHeight, ImgWidth, ImgChannels
        DetectedCircles = []
        for c in contours:
            # if cv2.contourArea(c) < 4000:
                # continue
            if len(c) < 3:
                continue
            Fit_Circle = CurveFitting.CircleFitting(img, c)
            if Fit_Circle is None:
                continue
            if (
                int(Fit_Circle[0][0]) < ImgWidth
                and int(Fit_Circle[0][1]) < ImgHeight
                and 6 < int(Fit_Circle[1]) < 300
            ):
                cir = (int(Fit_Circle[0][0]), int(
                    Fit_Circle[0][1]), int(Fit_Circle[1]))
                DetectedCircles.append(cir)

        return DetectedCircles

    @staticmethod
    def circleFiltering(DetectedCircles):
        ThresholdPixel = 4
        RequiredCircles = []
        UnwantedCircles = []

        for i in range(0, len(DetectedCircles)):
            c1 = DetectedCircles[i]
            c1 = (int(c1[0]), int(c1[1]), int(c1[2]))
            if c1 not in UnwantedCircles:
                if c1 in RequiredCircles:
                    continue
                if len(RequiredCircles) != 0:
                    IsInRequiredCircles = False
                    for j in RequiredCircles:
                        if (
                            fabs(j[0]-c1[0]) < ThresholdPixel
                            and fabs(j[1]-c1[1]) < ThresholdPixel
                            and fabs(j[2]-c1[2]) < ThresholdPixel
                        ):
                            IsInRequiredCircles = True
                            break
                    if IsInRequiredCircles:
                        continue
                    nc = (int(c1[0]), int(c1[1]), int(c1[2]))
                    RequiredCircles.append(nc)
                else:
                    nc = (int(c1[0]), int(c1[1]), int(c1[2]))
                    RequiredCircles.append(nc)

            for c in range(i+1, len(DetectedCircles)):
                c2 = DetectedCircles[c]
                c2 = (int(c2[0]), int(c2[1]), int(c2[2]))

                if c2 in UnwantedCircles:
                    continue
                if (
                    fabs(c1[0]-c2[0]) < ThresholdPixel
                    and fabs(c1[1]-c2[1]) < ThresholdPixel
                    and fabs(c1[2]-c2[2]) < ThresholdPixel
                ):
                    IsInRequiredCircles = False
                    if len(RequiredCircles) > 0:
                        for j in RequiredCircles:
                            if (
                                fabs(j[0]-c2[0]) < ThresholdPixel
                                and fabs(j[1]-c2[1]) < ThresholdPixel
                                and fabs(j[2]-c2[2]) < ThresholdPixel
                            ):
                                IsInRequiredCircles = True
                                break

                        if IsInRequiredCircles:
                            continue
                    nx = int((fabs(c1[0]+c2[0]))/2)
                    ny = int((fabs(c1[1]+c2[1]))/2)
                    nr = int((fabs(c1[2]+c2[2]))/2)
                    nc = (int(nx), int(ny), int(nr))
                    RequiredCircles.append(nc)
                    UnwantedCircles.append(c1)
                    UnwantedCircles.append(c2)
                else:
                    IsInRequiredCircles = False
                    for j in RequiredCircles:
                        if (
                            fabs(j[0]-c1[0]) < ThresholdPixel
                            and fabs(j[1]-c1[1]) < ThresholdPixel
                            and fabs(j[2]-c1[2]) < ThresholdPixel
                        ):
                            IsInRequiredCircles = True
                            break
                    if IsInRequiredCircles:
                        continue
                    nc = (int(c1[0]), int(c1[1]), int(c1[2]))
                    RequiredCircles.append(nc)
        RequiredCirclesExtracted = []
        for i in RequiredCircles:
            EC = ExtractedCircles()
            centre = Point2(i[0], i[1])
            radius = i[2]
            EC.ExtractCircle(EC, centre, radius)
            RequiredCirclesExtracted.append(EC)

        return RequiredCirclesExtracted

    @staticmethod
    def drawCircle(img, RequiredCircles):
        for i in RequiredCircles:
            c = i._centre
            r = i._radius
            cv2.circle(img, (int(c.x), int(c.y)), int(r), (0, 0, 255), 1)
        # cv2.imwrite(make_dir_root +"/Circle_Extraction_Output.png",img)
        return img

    @staticmethod
    def circleScanner(center, radius):
        switch = 3 - (2 * radius)
        points = set()
        x = 0
        y = radius
        while x <= y:
            points.add((x+center[0], -y+center[1]))
            points.add((y+center[0], -x+center[1]))
            points.add((y+center[0], x+center[1]))
            points.add((x+center[0], y+center[1]))
            points.add((-x+center[0], y+center[1]))
            points.add((-y+center[0], x+center[1]))
            points.add((-y+center[0], -x+center[1]))
            points.add((-x+center[0], -y+center[1]))
            if switch < 0:
                switch = switch + (4 * x) + 6
            else:
                switch = switch + (4 * (x - y)) + 10
                y = y - 1
            x = x + 1
        return points

    @staticmethod
    def CheckPixelsInVicinity(x, y, threshImg):
        scanRange = 2
        for i in range(-scanRange, scanRange):
            for j in range(-scanRange, scanRange):
                xj = x + j
                yi = y + i
                if threshImg[yi, xj] == 0:
                    return True
        return False

    @staticmethod
    def UniqueCircles(detectedcircles):
        global threshImg, ImgHeight, ImgWidth, ImgChannels
        DetectedCircles = []

        for i in detectedcircles:
            if not (i[0] <= ImgWidth and i[1] <= ImgHeight):
                continue
            points = CirclesFeature.circleScanner(
                (int(i[0]), int(i[1])), int(i[2])
            )
            total = len(points)
            nonzero = 0
            for p in points:
                if not (2 <= p[0] <= (ImgWidth - 2) and 2 <= p[1] <= (ImgHeight - 2)):
                    continue
                pixelpresent = CirclesFeature.CheckPixelsInVicinity(
                    p[0], p[1], threshImg
                )
                if pixelpresent == True:
                    nonzero += 1
            percent = (100 * nonzero) / total
            if percent < 50:
                continue
            c = (int(i[0]), int(i[1]), int(i[2]))
            DetectedCircles.append(c)
        return DetectedCircles


class CurveFitting():
    @staticmethod
    def CircleFitting(img, contour):
        ImgHeight, ImgWidth, ImgChannels = img.shape
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]
        fit = []
        loc = np.hstack([np.array(x).reshape(-1, 1),
                        np.array(y).reshape(-1, 1)])
        for ii in range(RANSAC_parameter['nIter']):
            randPoints = np.random.permutation(len(x))[:3]
            X = [x[randPoints[0]], x[randPoints[1]], x[randPoints[2]]]
            Y = [y[randPoints[0]], y[randPoints[1]], y[randPoints[2]]]
            (center, radius) = CurveFitting.fit_circle(X, Y)
            if not (
                RANSAC_parameter['lowerRadius'] < radius < RANSAC_parameter['upperRadius']
                and 0 <= center[0] < ImgWidth and 0 <= center[1] < ImgHeight
            ):
                continue
            centerDistance = np.linalg.norm(loc-center, axis=1)
            inCircle = np.where(
                np.abs(centerDistance-radius) < RANSAC_parameter['eps'])[0]
            inPts = len(inCircle)
            if (
                inPts < RANSAC_parameter['ransac_threshold'] *
                    4*np.pi*radius*RANSAC_parameter['eps']
                or inPts < 3
            ):
                continue

            xpt = x[inCircle]
            ypt = y[inCircle]
            (center, radius) = CurveFitting.fit_circle(xpt, ypt)
            fitC = (center, radius, inPts)
            fit.append(fitC)

        if len(fit) != 0:
            sortedFit = sorted(fit, key=lambda x: x[2], reverse=True)
            fitBestCircle = sortedFit[0]
        else:
            return None
        return fitBestCircle

    @staticmethod
    def calc_dist(x, y, xc, yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    @staticmethod
    def fit_circle(xPts, yPts):
        x_m = np.mean(xPts)
        y_m = np.mean(yPts)

        def calc_R(xc, yc):
            return np.sqrt((xPts-xc)**2 + (yPts-yc)**2)

        def f_2(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = x_m, y_m
        center, ier = optimize.leastsq(f_2, center_estimate)
        xc, yc = center
        Ri = calc_R(xc, yc)
        R = Ri.mean()
        return (center, R)
