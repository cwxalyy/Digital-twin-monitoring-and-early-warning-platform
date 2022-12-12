
import warnings
from venv import logger
from cv2.cv2 import CV_8UC3
import cv2.cv2 as cv
from ImageClass import Image
import numpy as np

# 处理数据格式
class StereoImage:
    """
    **SUMMARY**

    This class is for binaculor Stereopsis. That is exactrating 3D information from two differing views of a scene(Image). By comparing the two images, the relative depth information can be obtained.

    - Fundamental Matrix : F : a 3 x 3 numpy matrix, is a relationship between any two images of the same scene that constrains where the projection of points from the scene can occur in both images. see : http://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)

    - Homography Matrix : H : a 3 x 3 numpy matrix,

    - ptsLeft : The matched points on the left image.

    - ptsRight : The matched points on the right image.

    -findDisparityMap and findDepthMap - provides 3D information.

    for more information on stereo vision, visit : http://en.wikipedia.org/wiki/Computer_stereo_vision


    本课程适用于双筒立体视觉。 即从场景（图像）的两个不同视图中提取 3D 信息。 通过比较两幅图像，可以得到相对深度信息。

     - 基本矩阵：F：一个 3 x 3 numpy 矩阵，是同一场景的任何两个图像之间的关系，它限制了场景中点的投影可以出现在两个图像中的位置。 见：http://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)

     - 单应矩阵：H：一个 3 x 3 numpy 矩阵，

     - ptsLeft : 左侧图像上的匹配点。

     - ptsRight ：右侧图像上的匹配点。

     -findDisparityMap 和 findDepthMap - 提供 3D 信息。

     有关立体视觉的更多信息，请访问：http://en.wikipedia.org/wiki/Computer_stereo_vision
    **EXAMPLE**
    >>> img1 = Image('sampleimages/stereo_view1.png')
    >>> img2 = Image('sampleimages/stereo_view2.png')
    >>> stereoImg = StereoImage(img1,img2)
    >>> stereoImg.findDisparityMap(method="BM",nDisparity=20).show()
    """
    def __init__( self, imgLeft , imgRight ):
        self.ImageLeft = imgLeft
        self.ImageRight = imgRight
        if self.ImageLeft.size() != self.ImageRight.size():
            logger.warning('Left and Right images should have the same size.')
            return None
        else:
            self.size = self.ImageLeft.size()

    def get3DImage(self, Q, method="BM", state=None):
        """
        **SUMMARY**

        This method returns the 3D depth image using reprojectImageTo3D method.

        此方法使用 reprojectImageTo3D 方法返回 3D 深度图像。

        **PARAMETERS**

        * *Q* - reprojection Matrix (disparity to depth matrix)
        * *method* - Stereo Correspondonce method to be used.
                   - "BM" - Stereo BM
                   - "SGBM" - Stereo SGBM
        * *state* - dictionary corresponding to parameters of
                    stereo correspondonce.
                    SADWindowSize - odd int
                    nDisparity - int
                    minDisparity  - int
                    preFilterCap - int
                    preFilterType - int (only BM)
                    speckleRange - int
                    speckleWindowSize - int
                    P1 - int (only SGBM)
                    P2 - int (only SGBM)
                    fullDP - Bool (only SGBM)
                    uniquenessRatio - int
                    textureThreshold - int (only BM)

        **RETURNS**

        SimpleCV.Image representing 3D depth Image
        also StereoImage.Image3D gives OpenCV 3D Depth Image of CV_32F type.

        **EXAMPLE**

        >>> lImage = Image("l.jpg")
        >>> rImage = Image("r.jpg")
        >>> stereo = StereoImage(lImage, rImage)
        >>> Q = cv.Load("Q.yml")
        >>> stereo.get3DImage(Q).show()

        >>> state = {"SADWindowSize":9, "nDisparity":112, "minDisparity":-39}
        >>> stereo.get3DImage(Q, "BM", state).show()
        >>> stereo.get3DImage(Q, "SGBM", state).show()
        """
        imgLeft = self.ImageLeft
        imgRight = self.ImageRight
        cv2flag = True
        try:
            import cv2
        except ImportError:
            cv2flag = False
        import cv2 as cv
        (r, c) = self.size
        if method == "BM":
            sbm = cv.CreateStereoBMState()
            disparity = cv.CreateMat(c, r, cv.CV_32F)
            if state:
                SADWindowSize = state.get("SADWindowSize")
                preFilterCap = state.get("preFilterCap")
                minDisparity = state.get("minDisparity")
                numberOfDisparities = state.get("nDisparity")
                uniquenessRatio = state.get("uniquenessRatio")
                speckleRange = state.get("speckleRange")
                speckleWindowSize = state.get("speckleWindowSize")
                textureThreshold = state.get("textureThreshold")
                speckleRange = state.get("speckleRange")
                speckleWindowSize = state.get("speckleWindowSize")
                preFilterType = state.get("perFilterType")

                if SADWindowSize is not None:
                    sbm.SADWindowSize = SADWindowSize
                if preFilterCap is not None:
                    sbm.preFilterCap = preFilterCap
                if minDisparity is not None:
                    sbm.minDisparity = minDisparity
                if numberOfDisparities is not None:
                    sbm.numberOfDisparities = numberOfDisparities
                if uniquenessRatio is not None:
                    sbm.uniquenessRatio = uniquenessRatio
                if speckleRange is not None:
                    sbm.speckleRange = speckleRange
                if speckleWindowSize is not None:
                    sbm.speckleWindowSize = speckleWindowSize
                if textureThreshold is not None:
                    sbm.textureThreshold = textureThreshold
                if preFilterType is not None:
                    sbm.preFilterType = preFilterType
            else:
                sbm.SADWindowSize = 9
                sbm.preFilterType = 1
                sbm.preFilterSize = 5
                sbm.preFilterCap = 61
                sbm.minDisparity = -39
                sbm.numberOfDisparities = 112
                sbm.textureThreshold = 507
                sbm.uniquenessRatio= 0
                sbm.speckleRange = 8
                sbm.speckleWindowSize = 0

            gray_left = imgLeft.getGrayscaleMatrix()
            gray_right = imgRight.getGrayscaleMatrix()
            cv.FindStereoCorrespondenceBM(gray_left, gray_right, disparity, sbm)
            disparity_visual = cv.CreateMat(c, r, cv.CV_8U)

        elif method == "SGBM":
            if not cv2flag:
                warnings.warn("Can't Use SGBM without OpenCV >= 2.4. Use SBM instead.")
            sbm = cv2.StereoSGBM()

        if cv2flag:
            if not isinstance(Q, np.ndarray):
                Q = np.array(Q)
            if not isinstance(disparity, np.ndarray):
                disparity = np.array(disparity)
            Image3D = cv2.reprojectImageTo3D(disparity, Q, ddepth=cv2.cv.CV_32F)
            Image3D_normalize = cv2.normalize(Image3D, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8UC3)
            retVal = Image(Image3D_normalize, cv2image=True)
        else:
            Image3D = cv.CreateMat(self.LeftImage.size()[1], self.LeftImage.size()[0], cv2.cv.CV_32FC3)
            Image3D_normalize = cv.CreateMat(self.LeftImage.size()[1], self.LeftImage.size()[0], cv2.cv.CV_8UC3)
            cv.ReprojectImageTo3D(disparity, Image3D, Q)
            cv.Normalize(Image3D, Image3D_normalize, 0, 255, cv.CV_MINMAX, cv.CV_8UC3)
            retVal = Image(Image3D_normalize)
        self.Image3D = Image3D
        return retVal

    def get3DImageFromDisparity(self, disparity, Q):
        """
        **SUMMARY**

        This method returns the 3D depth image using reprojectImageTo3D method.

        **PARAMETERS**
        * *disparity* - Disparity Image
        * *Q* - reprojection Matrix (disparity to depth matrix)

        **RETURNS**

        SimpleCV.Image representing 3D depth Image
        also StereoCamera.Image3D gives OpenCV 3D Depth Image of CV_32F type.

        SimpleCV.Image 表示 3D 深度图像
        StereoCamera.Image3D 还提供了 CV_32F 类型的 OpenCV 3D 深度图像。
        **EXAMPLE**

        >>> lImage = Image("l.jpg")
        >>> rImage = Image("r.jpg")
        >>> stereo = StereoCamera()
        >>> Q = cv.Load("Q.yml")
        >>> disp = stereo.findDisparityMap()
        >>> stereo.get3DImageFromDisparity(disp, Q)
        """
        cv2flag = True
        try:
            import cv2
        except ImportError:
            cv2flag = False
            import cv2.cv as cv

        if cv2flag:
            if not isinstance(Q, np.ndarray):
                Q = np.array(Q)
            disparity = disparity.getNumpyCv2()
            Image3D = cv2.reprojectImageTo3D(disparity, Q, ddepth=cv2.cv.CV_32F)
            Image3D_normalize = cv2.normalize(Image3D, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8UC3)
            retVal = Image(Image3D_normalize, cv2image=True)
        else:
            disparity = disparity.getMatrix()
            Image3D = cv.CreateMat(self.LeftImage.size()[1], self.LeftImage.size()[0], cv2.cv.CV_32FC3)
            Image3D_normalize = cv.CreateMat(self.LeftImage.size()[1], self.LeftImage.size()[0], cv2.cv.CV_8UC3)
            cv.ReprojectImageTo3D(disparity, Image3D, Q)
            cv.Normalize(Image3D, Image3D_normalize, 0, 255, cv.CV_MINMAX, CV_8UC3)
            retVal = Image(Image3D_normalize)
        self.Image3D = Image3D
        return retVal
if __name__ == '__main__':
    lImage = Image("./images/1.png")
    rImage = Image("./outfile/1.png")
    stereo = StereoImage(lImage, rImage)
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])
    # Q = cv.Load("Q.yml")
    stereo.get3DImage(Q).show()

    state = {"SADWindowSize": 9, "nDisparity": 112, "minDisparity": -39}
    stereo.get3DImage(Q, "BM", state).show()
    stereo.get3DImage(Q, "SGBM", state).show()