'''
    Author: Dr. Lloyd Windrim
    Required packages: numpy, skimage, matplotlib

    Class for doing tree detection.

'''

# custom libraries
from forest3D import detection_tools, processLidar
from forest3D.object_detectors import detectObjects_yolov3 as detectObjects
from forest3D.IndexedPCD import IndexedPCD

# standard libaries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures as future
from skimage import exposure




class RasterDetector():

    def __init__(self, raster_layers, support_window=[1, 1, 1], normalisation='rescale', doHisteq=[True, True, True], res=[0.2, 0.2, 0.2], gridSize=[600, 600, 1000]):

        self.raster_layers = ['blank', 'blank', 'blank']

        for i, layer_name in enumerate(raster_layers):
            self.raster_layers[i] = layer_name

        self.support_window = [1, 1, 1]

        for i, val in enumerate(support_window):
            self.support_window[i] = val

        self.normalisation = normalisation

        self.doHisteq = [1, 1, 1]
        for i, val in enumerate(doHisteq):
            self.doHisteq[i] = val

        # Continuity with previous behaviour (supplying a single float)
        if isinstance(res, float):
            res = [res, res, res]

        self.res = res

        assert(len(gridSize) == 3)
        self.gridSize = np.array(gridSize)


    

    def sliding_training_data(self, detector_addr, xyz_data, colour_data=None, ground_pts=None, 
                              windowSize=[100, 100], stepSize=80, classID=0, confidence_thresh=0.5, 
                              overlap_thresh=5, spatialIndex="QuadTree"):

        if colour_data is None:
            xyz_clr_data = xyz_data
        elif len(np.shape(colour_data)) == 1:
            xyz_clr_data = np.hstack((xyz_data, colour_data[:, np.newaxis]))
        elif len(np.shape(colour_data)) == 2:
            xyz_clr_data = np.hstack((xyz_data, colour_data))

        xyz_clr_data = IndexedPCD(
            xyz_clr_data, [windowSize[0] - stepSize, windowSize[1] - stepSize], spatialIndex)
        aoi_list = detection_tools.create_all_windows(
            xyz_clr_data, stepSize, windowSize)

        img_store = []
        box_store = np.zeros((1, 4))
        ext_store = np.zeros((1, 6))
        totalCount = len(aoi_list)
        counter = 0

        for aoi in aoi_list:  # stepsize 100

            window = xyz_clr_data.points_in_aoi(aoi)

            # track progress
            counter = counter + 1
            sys.stdout.write("\r%d out of %d tiles" % (counter, totalCount))
            sys.stdout.flush()

            if len(window) > 0:

                raster_stack, centre = self._rasterise_quickly(
                    window, ground_pts=ground_pts)
                raster_stack = np.uint8(raster_stack*255)

                # use object detector to detect trees in raster
                [img, boxes, classes, scores] = detectObjects(
                    raster_stack,
                    addr_weights=os.path.join(detector_addr, 'yolov3.weights'),
                    addr_confg=os.path.join(detector_addr, 'yolov3.cfg'),
                    MIN_CONFIDENCE=confidence_thresh
                )

                if np.shape(boxes)[0] > 0:

                    # convert raster coordinates of bounding boxes to global x y coordinates
                    bb_coord = detection_tools.boundingBox_to_3dcoords(
                        boxes_=boxes,
                        gridSize_=self.gridSize[0:2],
                        gridRes_=self.res,
                        windowSize_=windowSize,
                        pcdCenter_=centre
                    )

                    # aggregate over windows
                    box_store = np.vstack((box_store, bb_coord[classes == classID, :]))
                    
                    img_store.append(img)
                    
                    ext = np.array(aoi.coords(), dtype = np.float64)
                    ext = np.concatenate((ext, centre))
                    ext_store = np.vstack((ext_store, ext))
        

        sys.stdout.write("\n")
        box_store = box_store[1:, :]
        ext_store = ext_store[1:, :]

        # remove overlapping boxes
        idx = detection_tools.find_unique_boxes2(box_store, overlap_thresh)
        box_store = box_store[idx, :]
        
        return box_store, img_store, ext_store




    def sliding_window_indexed(self, detector_addr, xyz_data, colour_data=None, ground_pts=None, windowSize=[100, 100], stepSize=80,
                               classID=0, confidence_thresh=0.5, overlap_thresh=5, returnBoxes=False,
                               spatialIndex="QuadTree"):

        if colour_data is None:
            xyz_clr_data = xyz_data
        elif len(np.shape(colour_data)) == 1:
            xyz_clr_data = np.hstack((xyz_data, colour_data[:, np.newaxis]))
        elif len(np.shape(colour_data)) == 2:
            xyz_clr_data = np.hstack((xyz_data, colour_data))

        xyz_clr_data = IndexedPCD(
            xyz_clr_data, [windowSize[0] - stepSize, windowSize[1] - stepSize], spatialIndex)
        aoi_list = detection_tools.create_all_windows(
            xyz_clr_data, stepSize, windowSize)

        box_store = np.zeros((1, 4))
        totalCount = len(aoi_list)
        counter = 0

        for aoi in aoi_list:  # stepsize 100

            window = xyz_clr_data.points_in_aoi(aoi)

            # track progress
            counter = counter + 1
            sys.stdout.write("\r%d out of %d tiles" % (counter, totalCount))
            sys.stdout.flush()

            if len(window) > 0:

                raster_stack, centre = self._rasterise_quickly(
                    window, ground_pts=ground_pts)
                raster_stack = np.uint8(raster_stack*255)

                # use object detector to detect trees in raster
                [img, boxes, classes, scores] = detectObjects(
                    raster_stack,
                    addr_weights=os.path.join(detector_addr, 'yolov3.weights'),
                    addr_confg=os.path.join(detector_addr, 'yolov3.cfg'),
                    MIN_CONFIDENCE=confidence_thresh
                )

                if np.shape(boxes)[0]:

                    # convert raster coordinates of bounding boxes to global x y coordinates
                    bb_coord = detection_tools.boundingBox_to_3dcoords(
                        boxes_=boxes,
                        gridSize_=self.gridSize[0:2],
                        gridRes_=self.res,
                        windowSize_=windowSize,
                        pcdCenter_=centre
                    )

                    # aggregate over windows
                    box_store = np.vstack(
                        (box_store, bb_coord[classes == classID, :]))

        sys.stdout.write("\n")
        box_store = box_store[1:, :]

        # remove overlapping boxes
        idx = detection_tools.find_unique_boxes2(box_store, overlap_thresh)
        box_store = box_store[idx, :]

        if returnBoxes:
            return box_store
        else:
            # label points in pcd according to which bounding box they are in.
            # return labels in the same order as the point-cloud provided.
            labels = detection_tools.label_pcd_from_bbox_indexed(
                xyz_clr_data, box_store[:, [1, 3, 0, 2]])
            return labels
    
    
    
    def sliding_window_future(self, detector_addr, xyz_data, colour_data=None, 
                              ground_pts=None, windowSize=[100, 100], stepSize=80,
                              classID=0, confidence_thresh=0.5, overlap_thresh=5, 
                              returnBoxes=False, max_cores = 1, 
                              spatialIndex="QuadTree"):

        if colour_data is None:
            xyz_clr_data = xyz_data
        elif len(np.shape(colour_data)) == 1:
            xyz_clr_data = np.hstack((xyz_data, colour_data[:, np.newaxis]))
        elif len(np.shape(colour_data)) == 2:
            xyz_clr_data = np.hstack((xyz_data, colour_data))

        xyz_clr_data = IndexedPCD(
            np_arr = xyz_clr_data, 
            gridSize = [windowSize[0] - stepSize, windowSize[1] - stepSize], 
            method = spatialIndex
        )
        
        aoi_list = detection_tools.create_all_windows(xyz_clr_data, stepSize, windowSize)


        # Process each window in parallel
        with future.ProcessPoolExecutor(max_workers = max_cores) as executor:
          
          # Initialise process class
          process_window = ProcessWindow(
            self, detector_addr, ground_pts, windowSize, 
            classID, confidence_thresh, overlap_thresh
          )
          
          # Iterate over the aoi_list
          boxes = executor.map(process_window, [xyz_clr_data.points_in_aoi(aoi) for aoi in aoi_list])
        
        
        counter = 0
        totalCount = len(aoi_list)
        box_store = np.zeros((1, 4))
        for box in boxes:
            
            # track progress
            counter = counter + 1
            sys.stdout.write("\r%d out of %d tiles" % (counter, totalCount))
            sys.stdout.flush()
            
            if (box is not None) and (box.shape[0] > 0):
                box_store = np.vstack((box_store, box))
        
        
        sys.stdout.write("\n")
        box_store = box_store[1:, :]

        # remove overlapping boxes
        idx = detection_tools.find_unique_boxes2(box_store, overlap_thresh)
        box_store = box_store[idx, :]

        if returnBoxes:
            return box_store
        else:
            # label points in pcd according to which bounding box they are in.
            labels = detection_tools.label_pcd_from_bbox_indexed(xyz_clr_data, box_store[:, [1, 3, 0, 2]])
            return labels
    
    


    def sliding_window(self, detector_addr, xyz_data, colour_data=None, ground_pts=None, windowSize=[100, 100], stepSize=80,
                       classID=0, confidence_thresh=0.5, overlap_thresh=5, returnBoxes=False):

        if colour_data is None:
            xyz_clr_data = xyz_data
        elif len(np.shape(colour_data)) == 1:
            xyz_clr_data = np.hstack((xyz_data, colour_data[:, np.newaxis]))
        elif len(np.shape(colour_data)) == 2:
            xyz_clr_data = np.hstack((xyz_data, colour_data))

        box_store = np.zeros((1, 4))
        counter = 0

        for (x, y, window) in detection_tools.sliding_window_3d(xyz_clr_data, stepSize=stepSize, windowSize=windowSize):  # stepsize 100

            # track progress
            counter = counter + 1
            totalCount = len(range(int(np.min(xyz_data[:, 0])), int(np.max(xyz_data[:, 0])), stepSize)) * \
                len(range(int(np.min(xyz_data[:, 1])), int(
                    np.max(xyz_data[:, 1])), stepSize))
            sys.stdout.write("\r%d out of %d tiles" % (counter, totalCount))
            sys.stdout.flush()

            if window is not None:

                raster_stack, centre = self._rasterise(
                    window, ground_pts=ground_pts)
                raster_stack = np.uint8(raster_stack*255)

                # use object detector to detect trees in raster
                [img, boxes, classes, scores] = detectObjects(raster_stack, addr_weights=os.path.join(detector_addr, 'yolov3.weights'),
                                                              addr_confg=os.path.join(detector_addr, 'yolov3.cfg'), MIN_CONFIDENCE=confidence_thresh)

                if np.shape(boxes)[0]:

                    # convert raster coordinates of bounding boxes to global x y coordinates
                    bb_coord = detection_tools.boundingBox_to_3dcoords(boxes_=boxes, gridSize_=self.gridSize[0:2], gridRes_=self.res,
                                                                       windowSize_=windowSize, pcdCenter_=centre)

                    # aggregate over windows
                    box_store = np.vstack(
                        (box_store, bb_coord[classes == classID, :]))

        sys.stdout.write("\n")
        box_store = box_store[1:, :]

        # remove overlapping boxes
        idx = detection_tools.find_unique_boxes2(box_store, overlap_thresh)
        box_store = box_store[idx, :]

        if returnBoxes:

            return box_store
        else:
            # label points in pcd according to which bounding box they are in
            labels = detection_tools.label_pcd_from_bbox(
                xyz_clr_data, box_store[:, [1, 3, 0, 2]])

            return labels

    def rasterise(self, xyz_data, colour_data=None, ground_pts=None, returnCentre=False):

        if colour_data is None:
            xyz_clr_data = xyz_data
        elif len(np.shape(colour_data)) == 1:
            xyz_clr_data = np.hstack((xyz_data, colour_data[:, np.newaxis]))
        elif len(np.shape(colour_data)) == 2:
            xyz_clr_data = np.hstack((xyz_data, colour_data))

        raster_stack, centre = self._rasterise(
            xyz_clr_data, ground_pts=ground_pts)

        if returnCentre:
            return raster_stack, centre
        else:
            return raster_stack

    def _rasterise(self, data, ground_pts=None):

        # create raster layers
        raster1, centre = get_raster(self.raster_layers[0], data.copy(), self.support_window[0], self.res,
                                     self.gridSize, ground_pts=ground_pts)
        raster2, _ = get_raster(self.raster_layers[1], data.copy(), self.support_window[1], self.res,
                                self.gridSize, ground_pts=ground_pts)
        raster3, _ = get_raster(self.raster_layers[2], data.copy(), self.support_window[2], self.res,
                                self.gridSize, ground_pts=ground_pts)

        # normalise
        if self.normalisation == 'rescale':

            rasters_eq = []
            for i, raster in enumerate([raster1, raster2, raster3]):
                # IDE complains about use of 'is not'
                if self.raster_layers[i] != 'blank':

                    plow, phigh = np.percentile(raster, (0, 100))
                    raster = exposure.rescale_intensity(
                        raster, in_range=(plow, phigh))
                    rasters_eq.append(raster)
                else:
                    rasters_eq.append(raster)
            raster_stack = np.stack(
                (rasters_eq[0], rasters_eq[1], rasters_eq[2]), axis=2)

        if self.normalisation == 'rescale+histeq':

            rasters_eq = []
            for i, raster in enumerate([raster1, raster2, raster3]):
                if self.raster_layers[i] != 'blank':

                    if self.doHisteq[i]:
                        raster_eq = exposure.equalize_hist(raster)
                    else:
                        raster_eq = raster
                    plow, phigh = np.percentile(raster_eq, (0, 100))
                    raster_eq = exposure.rescale_intensity(
                        raster_eq, in_range=(plow, phigh))
                    rasters_eq.append(raster_eq)
                else:
                    rasters_eq.append(raster)
            raster_stack = np.stack(
                (rasters_eq[0], rasters_eq[1], rasters_eq[2]), axis=2)

        elif self.normalisation == 'cmap_jet':

            assert ((self.raster_layers[1] == 'blank') & (
                self.raster_layers[2] == 'blank'))

            cmap = plt.cm.jet
            norm = plt.Normalize(vmin=raster1.min(), vmax=raster1.max())
            raster_stack = cmap(norm(raster1))[..., 0:3]

        return raster_stack, centre
    
    def _rasterise_quickly(self, data, ground_pts=None):


        # create occupancy grid representation here to avoid duplication
        PCD = processLidar.ProcessPC(data.copy()[:, :3])
        PCD.occupancyGrid_Binary(res_= self.res, gridSize_= self.gridSize)
        

        # create raster layers
        raster1, centre = get_raster_quicker(self.raster_layers[0], PCD, self.support_window[0], self.res,
                                             self.gridSize, ground_pts=ground_pts)
        raster2, _ = get_raster_quicker(self.raster_layers[1], PCD, self.support_window[1], self.res,
                                        self.gridSize, ground_pts=ground_pts)
        raster3, _ = get_raster_quicker(self.raster_layers[2], PCD, self.support_window[2], self.res,
                                        self.gridSize, ground_pts=ground_pts)

        # normalise
        if self.normalisation == 'rescale':

            rasters_eq = []
            for i, raster in enumerate([raster1, raster2, raster3]):
                # IDE complains about use of 'is not'
                if self.raster_layers[i] != 'blank':

                    plow, phigh = np.percentile(raster, (0, 100))
                    raster = exposure.rescale_intensity(
                        raster, in_range=(plow, phigh))
                    rasters_eq.append(raster)
                else:
                    rasters_eq.append(raster)
            raster_stack = np.stack(
                (rasters_eq[0], rasters_eq[1], rasters_eq[2]), axis=2)

        if self.normalisation == 'rescale+histeq':

            rasters_eq = []
            for i, raster in enumerate([raster1, raster2, raster3]):
                if self.raster_layers[i] != 'blank':

                    if self.doHisteq[i]:
                        raster_eq = exposure.equalize_hist(raster)
                    else:
                        raster_eq = raster
                    plow, phigh = np.percentile(raster_eq, (0, 100))
                    raster_eq = exposure.rescale_intensity(
                        raster_eq, in_range=(plow, phigh))
                    rasters_eq.append(raster_eq)
                else:
                    rasters_eq.append(raster)
            raster_stack = np.stack(
                (rasters_eq[0], rasters_eq[1], rasters_eq[2]), axis=2)

        elif self.normalisation == 'cmap_jet':

            assert ((self.raster_layers[1] == 'blank') & (
                self.raster_layers[2] == 'blank'))

            cmap = plt.cm.jet
            norm = plt.Normalize(vmin=raster1.min(), vmax=raster1.max())
            raster_stack = cmap(norm(raster1))[..., 0:3]

        return raster_stack, centre


def get_raster_quicker(method_name, PCD, support_window, res, gridSize, ground_pts=None):

    if method_name == 'vertical_density':
        PCD.vertical_density(support_window=support_window)
        
        return PCD.bev_verticalDensity, PCD._ProcessPC__centre[0:2]

    elif method_name == 'max_height':
        PCD.max_height(support_window=support_window)

        return PCD.bev_maxHeight, PCD._ProcessPC__centre[0:2]

    elif method_name == 'blank':

        return 0.5 * np.ones((gridSize[:2])), None



def get_raster(method_name, data, support_window, res, gridSize, ground_pts=None):

    if method_name == 'vertical_density':
        PCD = processLidar.ProcessPC(data[:, :3])
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.vertical_density(support_window=support_window)

        return PCD.bev_verticalDensity, PCD._ProcessPC__centre[0:2]

    elif method_name == 'mean_colour1':
        PCD = processLidar.ProcessPC(data[:, :3], pc_returns=(data[:, 3]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.mean_returns(support_window=support_window)

        return PCD.bev_meanReturn, PCD._ProcessPC__centre[0:2]

    elif method_name == 'mean_colour2':
        PCD = processLidar.ProcessPC(data[:, :3], pc_returns=(data[:, 4]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.mean_returns(support_window=support_window)

        return PCD.bev_meanReturn, PCD._ProcessPC__centre[0:2]

    elif method_name == 'mean_colour3':
        PCD = processLidar.ProcessPC(data[:, :3], pc_returns=(data[:, 5]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.mean_returns(support_window=support_window)

        return PCD.bev_meanReturn, PCD._ProcessPC__centre[0:2]

    elif method_name == 'max_colour1':
        PCD = processLidar.ProcessPC(data[:, :3], pc_returns=(data[:, 3]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.max_returns(support_window=support_window)

        return PCD.bev_maxReturn, PCD._ProcessPC__centre[0:2]

    elif method_name == 'max_colour2':
        PCD = processLidar.ProcessPC(data[:, :3], pc_returns=(data[:, 4]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.max_returns(support_window=support_window)

        return PCD.bev_maxReturn, PCD._ProcessPC__centre[0:2]

    elif method_name == 'max_colour3':
        PCD = processLidar.ProcessPC(data[:, :3], pc_returns=(data[:, 5]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.max_returns(support_window=support_window)

        return PCD.bev_maxReturn, PCD._ProcessPC__centre[0:2]

    elif method_name == 'max_height':
        PCD = processLidar.ProcessPC(data[:, :3])
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.max_height(support_window=support_window)

        return PCD.bev_maxHeight, PCD._ProcessPC__centre[0:2]

    elif method_name == 'canopy_height':
        PCD = processLidar.ProcessPC(data[:, :3])
        PCD.ground_normalise(ground_pts)
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.max_height(support_window=support_window)

        return PCD.bev_maxHeight, PCD._ProcessPC__centre[0:2]

    elif method_name == 'blank':

        return 0.5 * np.ones((gridSize[:2])), None


def pcd2rasterCoords(pts, gridSize, res, centre):
    # make sure to pass in pts with shape [numSamples x numCoordinates]
    # returns dictionary with 'col' and 'row' elements, which are 1d np.arrays

    if np.shape(res) == ():
        res = np.tile(res, (np.shape(pts)[1]))

    centred_pts = pts - centre[:np.shape(pts)[1]]

    # note: x point corresponds to row (first index) in the og grid -> therefore y (and vice-versa for y)
    coords = {}
    coords['row'] = np.array(np.clip(np.floor(
        (centred_pts[:, 0]-(-gridSize[0]/2.*res[0]))/res[0]), 0, gridSize[0]-1), dtype=int)  # y
    coords['col'] = np.array(np.clip(np.floor(
        (centred_pts[:, 1]-(-gridSize[1]/2.*res[1]))/res[1]), 0, gridSize[1]-1), dtype=int)  # x
    if np.shape(pts)[1] == 3:
        coords['z'] = np.array(np.clip(np.floor(
            (centred_pts[:, 2] - (-gridSize[2] / 2. * res[2])) / res[2]), 0, gridSize[2] - 1), dtype=int)
    return coords



## Since map has limitations on how arguments are passed to a method, it seemed
## easier to just instantiate a class with all the stuff that doesn't change
## from one sliding window to the next.
class ProcessWindow:
      
    def __init__(self, raster_detector, detector_addr, ground_pts = None, 
                windowSize = [100, 100], classID = 0, confidence_thresh = 0.5, 
                overlap_thresh = 5):
      
      self.raster_detector = raster_detector
      self.detector_addr = detector_addr
      self.ground_pts = ground_pts
      self.windowSize = windowSize
      self.classID = classID
      self.confidence_thresh = confidence_thresh
      self.overlap_thresh = overlap_thresh
      
      self.addr_weights = os.path.join(self.detector_addr, 'yolov3.weights')
      self.addr_config = os.path.join(self.detector_addr, 'yolov3.cfg')
    
    
    def __call__(self, window):
      
        if len(window) > 0:
          
          raster_stack,centre = self.raster_detector._rasterise_quickly(window, ground_pts = self.ground_pts)
          raster_stack = np.uint8(raster_stack * 255)
    
          # use object detector to detect trees in raster
          [img, boxes, classes, scores] = detectObjects(raster_stack, 
                                            addr_weights = self.addr_weights,
                                            addr_confg = self.addr_config, 
                                            MIN_CONFIDENCE = self.confidence_thresh
                                          )
    
          if np.shape(boxes)[0] == 0:
              return None
          
          
          # convert raster coordinates of bounding boxes to global x y coordinates
          bb_coord = detection_tools.boundingBox_to_3dcoords(
                      boxes_ = boxes, 
                      gridSize_ = self.raster_detector.gridSize[0:2], 
                      gridRes_ = self.raster_detector.res, 
                      windowSize_ = self.windowSize, 
                      pcdCenter_ = centre
                    )
          
        return bb_coord[classes == self.classID, :]
