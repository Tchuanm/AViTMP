
import os
import shutil

if __name__ == '__main__':

    path = "***/pytracking/tracking_results/avitmp/trackingnet/"
    path2 = "***/pytracking/tracking_results/avitmp/submit/"
    if not os.path.exists(path2):
        os.makedirs(path2)
    fileNames = os.listdir(path)
    for file in fileNames:
        sourceFilePath = path+file
        outFilePath = path2+file
        ifExists = os.path.exists(sourceFilePath)
        if not ifExists:
            print("Error:The Source file not exists!~-")
            sys.exit()

        with open(sourceFilePath , "r") as f:
            sourceFileStr = f.read()
        outFileStr = sourceFileStr.replace("\t",",")
        with open(outFilePath, "a+") as w:
            w.write(outFileStr)

    shutil.make_archive(path2, "zip", path2)
    shutil.rmtree(path2)
