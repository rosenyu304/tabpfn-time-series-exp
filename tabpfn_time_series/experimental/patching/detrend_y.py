import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Literal

import logging





def detrend_tsdf(tsdf, degree = 1):
    trends = []
    for item_id in tsdf.item_ids:
        values = np.array(tsdf.loc[item_id, 'target'])
        coeff_polynomial = np.polyfit(x=np.arange(len(values)), y=values, deg=degree)
        p = np.poly1d(coeff_polynomial)
        tsdf.loc[item_id, 'target'] = values - p(np.arange(len(values)))
        trend = p(np.arange(len(values)))
        trends.append(trend)
    return tsdf, trends


def retrend_tsdf(tsdf, trends, df_type="train"):
    index = 0
    tsdf = tsdf.copy()
    for item_id in tsdf.item_ids:
        
        if df_type == "train":
            values = np.array(tsdf.loc[item_id, 'target'])
            values = values + trends[index][:len(values)]
            tsdf.loc[item_id, 'target'] = values
            
        else:
            for col in tsdf.columns:
                values = np.array(tsdf.loc[item_id, col])
                values = values + trends[index][-len(values):]
                tsdf.loc[item_id, col] = values
        index += 1
        
    return tsdf