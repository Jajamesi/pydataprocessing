
import pandas as pd
from scipy.stats import  norm
import statsmodels.stats.proportion as proportion
import numpy as np

# sig 0.5
def z_test_prop(p1, p2, nobs1, nobs2, cl_sig):
    if (0.01 < p1 < 1) and (0.01 < p2 < 1):
        z_stat, p_value = proportion.proportions_ztest(
            np.array([p1*nobs1, p2*nobs2]), np.array([nobs1, nobs2]), alternative='two-sided')
        if p_value < cl_sig:
            if p1 > p2:
                return 1
            return 2

    return False


# sig 0.5
def z_mean_value(cl):
    z_two_tailed = norm.ppf(1 - cl / 2)
    return z_two_tailed


def z_test_mean(mean1, mean2, std_dev1 ,std_dev2, nobs1, nobs2, cl_sig_v):
    z = abs(mean1 - mean2) / (((std_dev1 ** 2) / nobs1 + (std_dev2 ** 2) / nobs2) ** (1 / 2))
    if z >= cl_sig_v:
        if mean1 > mean2:
            return 1
        return 2
    return False


def merge_sigs(perc, sigs):
    sigs.columns = pd.MultiIndex.from_tuples([
    (a, b, c, 'sig') for a, b, c, _ in perc.columns
    ])

    # Shortcuts
    cols1 = perc.columns.tolist()
    cols2 = sigs.columns.tolist()

    result_cols = [cols1[0]]

    for i in range(1, len(cols1)):
        result_cols.append(cols1[i])
        result_cols.append(cols2[i])

    result = pd.concat([perc, sigs], axis=1)[result_cols]

    return result


def calc_sig_prop(d, cl_sig):

    sig_df = d.copy()
    sig_df = sig_df.astype(str)
    sig_df[:] = np.nan

    # FOR EACH QUESTION
    total_col = ("Total","Total","T","t")

    for q, data in d.groupby(level=0):


        # из вопроса отобрали тотал и

        # total first
        base_name = data.index[-1]
        nobs1 = data.loc[base_name, total_col]
        if nobs1 > 29:
            for row_name in data.index:
                p1 = data.loc[row_name, total_col]

                for vs_col in data.columns[1:]:

                    p2 = data.loc[row_name, vs_col]
                    nobs2 = data.loc[base_name, vs_col]

                    if nobs2 > 29:

                        match z_test_prop(p1, p2, nobs1, nobs2, cl_sig):
                            case 1:
                                sig_df.loc[row_name, vs_col] = "<t"
                            case 2:
                                sig_df.loc[row_name, vs_col] = ">t"
                            case _:
                                pass
        else:
            sig_df.loc[(q,), total_col] = "_"


        # each banner vs inside
        for c, table in data.T.groupby(level=0):

            if table.columns[0]==total_col:
                continue

            table = table.T

            for col_i, column in enumerate(table.columns[:-1]):

                orig_l = column[3]

                versus_columns = table.columns[col_i+1:]

                base_name = table.index[-1]

                nobs1 = table.loc[base_name, column]

                if nobs1 > 29:

                    for row_name in table.index:
                        p1 = table.loc[row_name, column]

                        for vs_col in versus_columns:
                            vs_l = vs_col[3]
                            p2 = table.loc[row_name, vs_col]
                            nobs2 = table.loc[base_name, vs_col]

                            if nobs2 > 29:

                                match z_test_prop(p1, p2, nobs1, nobs2, cl_sig):
                                    case 1:
                                        vs_l_m = f">{vs_l}" if (sig_df.loc[row_name, column] == "<t") else vs_l
                                        sig_df.loc[row_name, column] = f">{vs_l}" if pd.isna(sig_df.loc[row_name, column]) else sig_df.loc[row_name, column] + vs_l_m
                                        # sig_df.loc[row_name, column] = sig_df.loc[row_name, column] + vs_l if (sig_df.loc[row_name, column] not in total_s) else f">{vs_l}"
                                    case 2:
                                        orig_l_m = f">{orig_l}" if (sig_df.loc[row_name, vs_col] == "<t") else orig_l
                                        sig_df.loc[row_name, vs_col] = f">{orig_l}" if pd.isna(sig_df.loc[row_name, vs_col]) else sig_df.loc[row_name, vs_col] + orig_l_m
                                        # sig_df.loc[row_name, vs_col] = sig_df.loc[row_name, vs_col] + orig_l if (sig_df.loc[row_name, vs_col] not in total_s) else f">{orig_l}"
                                    case _:
                                        pass
                else:
                    sig_df.loc[(q,), column] = "_"

    return merge_sigs(d, sig_df)



def calc_sig_mean(d, cl_sig):

    mean_s ="Среднее"
    std_s ="Среднеквадратичное отклонение"
    base_s ="Валидные"

    cl_sig_v = z_mean_value(cl_sig)
    print(cl_sig_v)

    sig_df = d.copy()
    sig_df = sig_df.astype(str)
    sig_df[:] = np.nan

    # FOR EACH QUESTION
    total_col = ("Total","Total","T","t")

    for q, data in d.groupby(level=0):
        # из вопроса отобрали тотал и

        # total first
        mean1 = data.loc[(q, mean_s), total_col]
        std_dev1 = data.loc[(q, std_s), total_col]
        nobs1 = data.loc[(q, base_s), total_col]
        if nobs1 > 29:



            for vs_col in data.columns[1:]:

                mean2 = data.loc[(q, mean_s), vs_col]
                std_dev2 = data.loc[(q, std_s), vs_col]
                nobs2 = data.loc[(q, base_s), vs_col]

                if nobs2 > 29:

                    match z_test_mean(mean1, mean2, std_dev1, std_dev2, nobs1, nobs2, cl_sig_v):
                        case 1:
                            sig_df.loc[(q, mean_s), vs_col] = "<t"
                        case 2:
                            sig_df.loc[(q, mean_s), vs_col] = ">t"
                        case _:
                            pass
        else:
            sig_df.loc[(q,), total_col] = "_"


        # each banner vs inside
        for c, table in data.T.groupby(level=0):

            if table.columns[0]==total_col:
                continue

            table = table.T

            for col_i, column in enumerate(table.columns[:-1]):

                orig_l = column[3]

                versus_columns = table.columns[col_i+1:]

                mean1 = data.loc[(q, mean_s), column]
                std_dev1 = data.loc[(q, std_s), column]
                nobs1 = data.loc[(q, base_s), column]

                if nobs1 > 29:

                    for vs_col in versus_columns:
                        mean2 = data.loc[(q, mean_s), vs_col]
                        std_dev2 = data.loc[(q, std_s), vs_col]
                        nobs2 = data.loc[(q, base_s), vs_col]

                        vs_l = vs_col[3]

                        if nobs2 > 29:

                            match z_test_mean(mean1, mean2, std_dev1, std_dev2, nobs1, nobs2, cl_sig_v):
                                case 1:
                                    vs_l_m = f">{vs_l}" if (sig_df.loc[(q, mean_s), column] == "<t") else vs_l
                                    sig_df.loc[(q, mean_s), column] = f">{vs_l}" if pd.isna(sig_df.loc[(q, mean_s), column]) else sig_df.loc[(q, mean_s), column] + vs_l_m
                                    # sig_df.loc[row_name, column] = sig_df.loc[row_name, column] + vs_l if (sig_df.loc[row_name, column] not in total_s) else f">{vs_l}"
                                case 2:
                                    orig_l_m = f">{orig_l}" if (sig_df.loc[(q, mean_s), vs_col] == "<t") else orig_l
                                    sig_df.loc[(q, mean_s), vs_col] = f">{orig_l}" if pd.isna(sig_df.loc[(q, mean_s), vs_col]) else sig_df.loc[(q, mean_s), vs_col] + orig_l_m
                                    # sig_df.loc[row_name, vs_col] = sig_df.loc[row_name, vs_col] + orig_l if (sig_df.loc[row_name, vs_col] not in total_s) else f">{orig_l}"
                                case _:
                                    pass
                else:
                    sig_df.loc[(q,), column] = "_"

    return merge_sigs(d, sig_df)