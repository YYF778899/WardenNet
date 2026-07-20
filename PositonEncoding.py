import os
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

def bivariate_gaussian(coords, A, x0, y0, sigma_x, sigma_y, C):
    x, y = coords

    eps = 1e-8
    exponent = -((x - x0) ** 2 / (2 * (sigma_x ** 2 + eps)) + (y - y0) ** 2 / (2 * (sigma_y ** 2 + eps)))
    return A * np.exp(exponent) + C



def fit_bivariate_gaussian(x_data, y_data, z_data):

    coords = np.vstack((x_data, y_data))

    idx_max = np.argmax(z_data)
    x0_init = x_data[idx_max]
    y0_init = y_data[idx_max]
    A_init = np.max(z_data) - np.min(z_data)
    C_init = np.min(z_data)

    #
    sigma_x_init = np.std(x_data) if np.std(x_data) > 0 else 1.0
    sigma_y_init = np.std(y_data) if np.std(y_data) > 0 else 1.0

    p0 = [A_init, x0_init, y0_init, sigma_x_init, sigma_y_init, C_init]

    #
    lower_bounds = [-np.inf, -np.inf, -np.inf, 1e-5, 1e-5, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    bounds = (lower_bounds, upper_bounds)

    #
    popt, _ = curve_fit(bivariate_gaussian, coords, z_data, p0=p0, bounds=bounds, maxfev=10000)

    return popt


if __name__ == '__main__':


        TabelListX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        TabelListY = [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        newList = []
        for i in TabelListY:
            for j in TabelListX:
                newList.append((j, i))
        new_coordinates = np.array(newList)
        grid_x = new_coordinates[:, 0]
        grid_y = new_coordinates[:, 1]
        grid_coords = np.vstack((grid_x, grid_y))

        out_path = '/Save/20-20-Data'

        file_path = '/.../excel.xlsx'
        nameList = ['Sample1']

        for name in nameList:

            data = pd.read_excel(file_path, sheet_name=name)

            x_coords = data['X'].values
            y_coords = data['Y'].values
            values1 = data['CA153'].values
            values2 = data['CA125'].values
            values3 = data['CEA'].values

            # 检查样本点数量是否满足描述中“至少6个采样点”的要求
            if len(x_coords) < 6:
                print(f"警告:  {name} less 6，pass")
                continue

            try:
                #
                popt1 = fit_bivariate_gaussian(x_coords, y_coords, values1)
                popt2 = fit_bivariate_gaussian(x_coords, y_coords, values2)
                popt3 = fit_bivariate_gaussian(x_coords, y_coords, values3)

                #
                pred_grid1 = bivariate_gaussian(grid_coords, *popt1).reshape((20, 20))
                pred_grid2 = bivariate_gaussian(grid_coords, *popt2).reshape((20, 20))
                pred_grid3 = bivariate_gaussian(grid_coords, *popt3).reshape((20, 20))

                # 1, 20, 20
                newNpy1 = np.expand_dims(pred_grid1, axis=0)
                newNpy2 = np.expand_dims(pred_grid2, axis=0)
                newNpy3 = np.expand_dims(pred_grid3, axis=0)

                # 3, 20, 20
                finalArray = np.concatenate([newNpy1, newNpy2, newNpy3], axis=0)

                # 保存为 .npy 文件
                output_file = os.path.join(out_path, f"{name}.npy")
                np.save(output_file, finalArray)
                print(f"Save: {output_file}")

            except Exception as e:
                print(f" {name} Error: {e}")
