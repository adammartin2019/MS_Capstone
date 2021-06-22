#import necessary libraries
import arcpy
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
print("Libraries successfully imported")


#PreProcessing
print("beginning preprocessing steps...")

#initialize input variables
DB_Path = "C:/Users/Adam Martin/Desktop/School Files/UMD Files/GraduateSchool/MSGIS/GEOG797_Capstone/Data/Cleaned_Data.gdb"
CTX6m = arcpy.Raster(DB_Path+"/CTX_6m_georef_8bit")
CTX_DTM = arcpy.Raster(DB_Path+"/CTX_DTM_georef_8bit")
THEMIS_Day = arcpy.Raster(DB_Path+"/THEMIS_Day_georef_8bit")
THEMIS_Night = arcpy.Raster(DB_Path+"/THEMIS_Night_georef_8bit")
THEMIS_NightOverDay = arcpy.Raster(DB_Path+"/THEMIS_NightOverDay_georef_8bit")
THEMIS_DCS = arcpy.Raster(DB_Path+"/THEMIS_DCS_georef_8bit")
print("Raster variables initialized")



#convert rasters to numpy arrays
CTX6m_arr = arcpy.RasterToNumPyArray(CTX6m, nodata_to_value=0)
CTX_DTM_arr = arcpy.RasterToNumPyArray(CTX_DTM, nodata_to_value=0)
THEMIS_Day_arr = arcpy.RasterToNumPyArray(THEMIS_Day, nodata_to_value=0)
THEMIS_Night_arr = arcpy.RasterToNumPyArray(THEMIS_Night, nodata_to_value=0)
THEMIS_NightOverDay_arr = arcpy.RasterToNumPyArray(THEMIS_NightOverDay, nodata_to_value=0)
THEMIS_DCS_arr = arcpy.RasterToNumPyArray(THEMIS_DCS, nodata_to_value=0)
print("Converted rasters to numpy arrays")



#trim and reshape arrays, extend 0 axis, flatten and concat data
CTX6m_trim = CTX6m_arr[:5040, :5320]
CTX_DTM_trim = CTX_DTM_arr[:5040, :5320]
THEMIS_DAY_trim = THEMIS_Day_arr[:,:5040, :5320]
THEMIS_NIGHT_trim = THEMIS_Night_arr[:,:5040, :5320]
THEMIS_NIGHTOVERDAY_trim = THEMIS_NightOverDay_arr[:,:5040,:5320]
THEMIS_DCS_trim = THEMIS_DCS_arr[:,:5040, :5320]
print("all arrays trimmed")


#expand the 2d arrays
CTX6m_trim_expanded = np.expand_dims(CTX6m_trim, axis=0)

CTX_DTM_expanded = np.expand_dims(CTX_DTM_trim, axis=0)
print("CTX arrays expanded")



## concatenate arrays for different inputs

#CTX only
inData_CTX = np.reshape(CTX6m_trim_expanded, [CTX6m_trim_expanded.shape[0], 
                                              CTX6m_trim_expanded.shape[1]*CTX6m_trim_expanded.shape[2]])
print("CTX flattened array shape:", inData_CTX.shape)



#DTM only
inData_DTM = np.reshape(CTX_DTM_expanded, [CTX_DTM_expanded.shape[0], 
                                              CTX_DTM_expanded.shape[1]*CTX_DTM_expanded.shape[2]])
print("DTM flattened array shape:", inData_DTM.shape)



#NightOverDay only
inData_NoD = np.reshape(THEMIS_NIGHTOVERDAY_trim, [THEMIS_NIGHTOVERDAY_trim.shape[0], 
                                              THEMIS_NIGHTOVERDAY_trim.shape[1]*THEMIS_NIGHTOVERDAY_trim.shape[2]])
print("NightOverDay flattened array shape:", inData_NoD.shape)




#DCSonly
inData_DCS = np.reshape(THEMIS_DCS_trim, [THEMIS_DCS_trim.shape[0], 
                                              THEMIS_DCS_trim.shape[1]*THEMIS_DCS_trim.shape[2]])
print("DCS flattened array shape:", inData_DCS.shape)



#CTX and DTM combined
Data_CTX_DTM = np.concatenate([CTX6m_trim_expanded,CTX_DTM_expanded], axis=0)
print("CTX_DTM concatenated array shape:", Data_CTX_DTM.shape)

inData_CTX_DTM = np.reshape(Data_CTX_DTM, [Data_CTX_DTM.shape[0], 
                                           Data_CTX_DTM.shape[1]*Data_CTX_DTM.shape[2]])
print("CTX_DTM flattened array shape:", inData_CTX_DTM.shape)



#CTX DTM and DCS combined
Data_CTX_DTM_DCS = np.concatenate([CTX6m_trim_expanded,CTX_DTM_expanded,THEMIS_DCS_trim], axis=0)
print("CTX_DTM_DCS concatenated array shape:", Data_CTX_DTM_DCS.shape)

inData_CTX_DTM_DCS = np.reshape(Data_CTX_DTM_DCS, [Data_CTX_DTM_DCS.shape[0], 
                                           Data_CTX_DTM_DCS.shape[1]*Data_CTX_DTM_DCS.shape[2]])
print("CTX_DTM_DCS flattened array shape:", inData_CTX_DTM_DCS.shape)




#CTX DTM and NoD combined
Data_CTX_DTM_NoD = np.concatenate([CTX6m_trim_expanded,CTX_DTM_expanded, THEMIS_NIGHTOVERDAY_trim], axis=0)
print("CTX_DTM_NoD concatenated array shape:", Data_CTX_DTM_NoD.shape)

inData_CTX_DTM_NoD = np.reshape(Data_CTX_DTM_NoD, [Data_CTX_DTM_NoD.shape[0], 
                                           Data_CTX_DTM_NoD.shape[1]*Data_CTX_DTM_NoD.shape[2]])
print("CTX_DTM_NoD flattened array shape:", inData_CTX_DTM_NoD.shape)




#CTX NoD and DCS combined
Data_CTX_NoD_DCS = np.concatenate([CTX6m_trim_expanded,THEMIS_NIGHTOVERDAY_trim, THEMIS_DCS_trim], axis=0)
print("CTX_NoD_DCS concatenated array shape:", Data_CTX_NoD_DCS.shape)

inData_CTX_NoD_DCS = np.reshape(Data_CTX_NoD_DCS, [Data_CTX_NoD_DCS.shape[0], 
                                           Data_CTX_NoD_DCS.shape[1]*Data_CTX_NoD_DCS.shape[2]])
print("CTX_NoD_DCS flattened array shape:", inData_CTX_NoD_DCS.shape)




#All data sets combined
X_all = np.concatenate([CTX6m_trim_expanded, 
                        CTX_DTM_expanded,
                        THEMIS_DAY_trim,
                        THEMIS_NIGHT_trim,
                        THEMIS_NIGHTOVERDAY_trim,
                        THEMIS_DCS_trim
                        ], axis=0)
print("All data layers concatenated array shape:", X_all.shape)

## flatten the concatenated array - can reuse the same variable name
X_all = np.reshape(X_all, [X_all.shape[0], X_all.shape[1]*X_all.shape[2]])
print("All data layersflattened array shape:", X_all.shape)
## flattened array should have shape [n_bands, 5040*5320]


print("All input data prepared")


#Delete unnecessary variables to free up memory
del CTX6m 
del CTX_DTM 
del THEMIS_Day 
del THEMIS_Night 
del THEMIS_NightOverDay 
del THEMIS_DCS
del CTX6m_arr
del CTX_DTM_arr
del THEMIS_Day_arr
del THEMIS_Night_arr
del THEMIS_NightOverDay_arr
del THEMIS_DCS_arr
del CTX6m_trim 
del CTX_DTM_trim 
del THEMIS_DAY_trim 
del THEMIS_NIGHT_trim 
del THEMIS_NIGHTOVERDAY_trim 
del THEMIS_DCS_trim 
del CTX6m_trim_expanded
del CTX_DTM_expanded
gc.collect()
print("garbage collected.")



#KMeans and image creation

## kmeans expects the data to be shape N x M where 
## N is the number of samples (pixels) and M is the 
## number of features (bands) - so need to move the axis

##Change out the input feature set based on which kmeans needs to be done

Kmeans_In_Data = np.moveaxis(inData_CTX_DTM_NoD, 0, 1) 
print("input data shape for kmeans:", Kmeans_In_Data.shape)
## should now have shape [5040*5320, n_bands]

#Run kmeans, change fit variable for different layer results
print("kmeans starting....")
kmeans = KMeans(n_clusters=15).fit(Kmeans_In_Data)
print("Kmeans finished running")


## kmeans.labels_ should have shape [5040*5320]
## need to reshape to original image dimension
LabelArr = np.reshape(kmeans.labels_, [5040, 5320])
print(LabelArr.shape)
print("Finished reshaping kmeans label array")

#convert the final kmeans labels array back to a raster for rendering
Kmeans_raster = arcpy.NumPyArrayToRaster(LabelArr,x_cell_size=6)
print("array converted back to raster")

print("All done!")

