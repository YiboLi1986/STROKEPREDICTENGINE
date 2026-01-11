import os
import joblib
import pandas as pd
import tensorflow as tf

from xgboost import XGBClassifier

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


class VotingModels:
    def __init__(self, input: dict, threshold=0.5):
        self.ml_model = tf.keras.models.load_model('../models/Full_features_with_label/ML_model_for_stroke_prediction.h5')
        self.rf_model = joblib.load('../models/Full_features_with_label/RF_model_for_stroke_prediction.pkl')
        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model('../models/Full_features_with_label/XGB_model_for_stroke_prediction.json')

        self.input = input
        self.df = pd.DataFrame(self.input, index=[0])

        self.threshold = threshold

    def voting(self):
        y_pred_ml = self.ml_model.predict(self.df)
        y_pred_rf = self.rf_model.predict_proba(self.df)
        y_pred_xgb = self.xgb_model.predict_proba(self.df)

        # Print out the prediction probabilities for each class
        print("Model of ML_model_for_stroke_prediction result: " +
              "the predict probability as non-stroke (0) is: " + str(y_pred_ml[0][0]) +
              "; the predict probability as stroke (1) is: " + str(1 - y_pred_ml[0][0]))

        print("Model of RF_model_for_stroke_prediction result: " +
              "the predict probability as non-stroke (0) is: " + str(y_pred_rf[0][0]) +
              "; the predict probability as stroke (1) is: " + str(y_pred_rf[0][1]))

        print("Model of XGB_model_for_stroke_prediction result: " +
              "the predict probability as non-stroke (0) is: " + str(y_pred_xgb[0][0]) +
              "; the predict probability as stroke (1) is: " + str(y_pred_xgb[0][1]))

        print("*" * 100)

        if y_pred_ml[0][0] > self.threshold:
            res_ml = "non-stroke"
        elif (1 - y_pred_ml[0][0]) > self.threshold:
            res_ml = "stroke"
        else:
            res_ml = "uncertainty"
        print("Model of ML_model_for_stroke_prediction indicates this is a " + res_ml + " case!")

        if y_pred_rf[0][0] > self.threshold:
            res_rf = "non-stroke"
        elif y_pred_rf[0][1] > self.threshold:
            res_rf = "stroke"
        else:
            res_rf = "uncertainty"
        print("Model of RF_model_for_stroke_prediction indicates this is a " + res_rf + " case!")

        if y_pred_xgb[0][0] > self.threshold:
            res_xgb = "non-stroke"
        elif y_pred_xgb[0][1] > self.threshold:
            res_xgb = "stroke"
        else:
            res_xgb = "uncertainty"
        print("Model of XGB_model_for_stroke_prediction indicates this is a " + res_xgb + " case!")


class VotingModels_2:
    def __init__(self, input: dict, threshold=0.5):
        self.ml_model = None
        self.rf_model = None
        self.xgb_model = None

        self.input = {k: v for k, v in input.items() if v is not None}
        self.df = pd.DataFrame(self.input, index=[0])
        print("DataFrame content:")
        print(self.df)

        self.threshold = threshold

    def extract_keys(self, input: dict):
        return [key for key, value in input.items() if value is not None]

    def process_subfolder_names(self, folder_path, feature_list):
        matched = False

        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)

            if os.path.isdir(subfolder_path):
                name_parts = subfolder_name.split('_')
                processed_name = '_'.join(name_parts[:-2])
                processed_parts = processed_name.split('_')

                if sorted(processed_parts) == sorted(feature_list):
                    matched = True

                    try:
                        ml_model_path = os.path.join(subfolder_path, 'ML_model_for_stroke_prediction.h5')
                        rf_model_path = os.path.join(subfolder_path, 'RF_model_for_stroke_prediction.pkl')
                        xgb_model_path = os.path.join(subfolder_path, 'XGB_model_for_stroke_prediction.json')

                        if os.path.exists(ml_model_path):
                            self.ml_model = tf.keras.models.load_model(ml_model_path)
                        if os.path.exists(rf_model_path):
                            self.rf_model = joblib.load(rf_model_path)
                        if os.path.exists(xgb_model_path):
                            self.xgb_model = XGBClassifier()
                            self.xgb_model.load_model(xgb_model_path)
                    except Exception as e:
                        print(f"Error loading models: {e}")

                    break

        return matched

    def voting(self):
        y_pred_ml = self.ml_model.predict(self.df)
        y_pred_rf = self.rf_model.predict_proba(self.df)
        y_pred_xgb = self.xgb_model.predict_proba(self.df)

        # Print out the prediction probabilities for each class
        print("Model of ML_model_for_stroke_prediction result: " +
              "the predict probability as non-stroke (0) is: " + str(y_pred_ml[0][0]) +
              "; the predict probability as stroke (1) is: " + str(1 - y_pred_ml[0][0]))

        print("Model of RF_model_for_stroke_prediction result: " +
              "the predict probability as non-stroke (0) is: " + str(y_pred_rf[0][0]) +
              "; the predict probability as stroke (1) is: " + str(y_pred_rf[0][1]))

        print("Model of XGB_model_for_stroke_prediction result: " +
              "the predict probability as non-stroke (0) is: " + str(y_pred_xgb[0][0]) +
              "; the predict probability as stroke (1) is: " + str(y_pred_xgb[0][1]))

        print("*" * 100)

        if y_pred_ml[0][0] > self.threshold:
            res_ml = "non-stroke"
        elif (1 - y_pred_ml[0][0]) > self.threshold:
            res_ml = "stroke"
        else:
            res_ml = "uncertainty"
        print("Model of ML_model_for_stroke_prediction indicates this is a " + res_ml + " case!")

        if y_pred_rf[0][0] > self.threshold:
            res_rf = "non-stroke"
        elif y_pred_rf[0][1] > self.threshold:
            res_rf = "stroke"
        else:
            res_rf = "uncertainty"
        print("Model of RF_model_for_stroke_prediction indicates this is a " + res_rf + " case!")

        if y_pred_xgb[0][0] > self.threshold:
            res_xgb = "non-stroke"
        elif y_pred_xgb[0][1] > self.threshold:
            res_xgb = "stroke"
        else:
            res_xgb = "uncertainty"
        print("Model of XGB_model_for_stroke_prediction indicates this is a " + res_xgb + " case!")


def is_empty(x_test):
    missing_list = []
    for key,value in x_test.items():
        if value is None:
            missing_list.append(key)
    return missing_list


def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False

"""
    race: val/10
    sex: val/10
    HTN: val/10
    DM: val/10
    HLD: val/10
    Smoking: val/10
    HxOfStroke: val/10
    HxOfSeizure: val/10
    FacialDroop: val/10
"""


def processInput():
    x_test = {'race': 0, 'sex': 0, 'HTN': 0, 'DM': 0, 'HLD': 0, 'Smoking': 0, 'HxOfStroke': 0, 'HxOfSeizure': 0, 'FacialDroop': 0}

    for name in x_test.keys():
        if name == 'race':
            while True:
                print("*" * 30 + " Please enter the \033[1;32;1m {} \033[0m (1-C, 2-AA, 3-You don't know) ".format(name) + "*" * 30)
                val = input("Enter: ")
                print("The value you enter is: " + val)
                if val.strip() == "" or (is_number(val) and float(val) == 3):
                    x_test[name] = None
                    break
                elif is_number(val) and (float(val) == 1 or float(val) == 2):
                    x_test[name] = float(val) / 10
                    break
                else:
                    print("The value you enter is out of range or the value is illegal character, please enter again: ")

        elif name == 'sex':
            while True:
                print("*" * 30 + " Please enter the \033[1;32;1m {} \033[0m (1-male, 2-female, 3-You don't know) ".format(name) + "*" * 30)
                val = input("Enter: ")
                print("The value you enter is: " + val)
                if val.strip() == "" or (is_number(val) and float(val) == 3):
                    x_test[name] = None
                    break
                elif is_number(val) and (float(val) == 1 or float(val) == 2):
                    x_test[name] = float(val) / 10
                    break
                else:
                    print("The value you enter is out of range or the value is illegal character, please enter again: ")

        elif name == 'HTN':
            while True:
                print("*" * 30 + " Please enter the \033[1;32;1m {} \033[0m (0-no, 1-yes, 2-You don't know) ".format(name) + "*" * 30)
                val = input("Enter: ")
                print("The value you enter is: " + val)
                if val.strip() == "" or (is_number(val) and float(val) == 2):
                    x_test[name] = None
                    break
                elif is_number(val) and (float(val) == 0 or float(val) == 1):
                    x_test[name] = float(val) / 10
                    break
                else:
                    print("The value you enter is out of range or the value is illegal character, please enter again: ")

        elif name == 'DM':
            while True:
                print("*" * 30 + " Please enter the \033[1;32;1m {} \033[0m (0-no, 1-yes, 2-You don't know) ".format(name) + "*" * 30)
                val = input("Enter: ")
                print("The value you enter is: " + val)
                if val.strip() == "" or (is_number(val) and float(val) == 2):
                    x_test[name] = None
                    break
                elif is_number(val) and (float(val) == 0 or float(val) == 1):
                    x_test[name] = float(val) / 10
                    break
                else:
                    print("The value you enter is out of range or the value is illegal character, please enter again: ")

        elif name == 'HLD':
            while True:
                print("*" * 30 + " Please enter the \033[1;32;1m {} \033[0m (0-no, 1-yes, 2-You don't know) ".format(name) + "*" * 30)
                val = input("Enter: ")
                print("The value you enter is: " + val)
                if val.strip() == "" or (is_number(val) and float(val) == 2):
                    x_test[name] = None
                    break
                elif is_number(val) and (float(val) == 0 or float(val) == 1):
                    x_test[name] = float(val) / 10
                    break
                else:
                    print("The value you enter is out of range or the value is illegal character, please enter again: ")

        elif name == 'Smoking':
            while True:
                print("*" * 30 + " Please enter the \033[1;32;1m {} \033[0m (0-no, 1-yes, 2-You don't know) ".format(name) + "*" * 30)
                val = input("Enter: ")
                print("The value you enter is: " + val)
                if val.strip() == "" or (is_number(val) and float(val) == 2):
                    x_test[name] = None
                    break
                elif is_number(val) and (float(val) == 0 or float(val) == 1):
                    x_test[name] = float(val) / 10
                    break
                else:
                    print("The value you enter is out of range or the value is illegal character, please enter again: ")

        elif name == 'HxOfStroke':
            while True:
                print("*" * 30 + " Please enter the \033[1;32;1m {} \033[0m (0-no, 1-yes, 2-You don't know) ".format(name) + "*" * 30)
                val = input("Enter: ")
                print("The value you enter is: " + val)
                if val.strip() == "" or (is_number(val) and float(val) == 2):
                    x_test[name] = None
                    break
                elif is_number(val) and (float(val) == 0 or float(val) == 1):
                    x_test[name] = float(val) / 10
                    break
                else:
                    print("The value you enter is out of range or the value is illegal character, please enter again: ")

        elif name == 'HxOfSeizure':
            while True:
                print("*" * 30 + " Please enter the \033[1;32;1m {} \033[0m (0-no, 1-yes, 2-You don't know) ".format(name) + "*" * 30)
                val = input("Enter: ")
                print("The value you enter is: " + val)
                if val.strip() == "" or (is_number(val) and float(val) == 2):
                    x_test[name] = None
                    break
                elif is_number(val) and (float(val) == 0 or float(val) == 1):
                    x_test[name] = float(val) / 10
                    break
                else:
                    print("The value you enter is out of range or the value is illegal character, please enter again: ")

        elif name == 'FacialDroop':
            while True:
                print("*" * 30 + " Please enter the \033[1;32;1m {} \033[0m (0-no, 1-yes, 2-You don't know) ".format(name) + "*" * 30)
                val = input("Enter: ")
                print("The value you enter is: " + val)
                if val.strip() == "" or (is_number(val) and float(val) == 2):
                    x_test[name] = None
                    break
                elif is_number(val) and (float(val) == 0 or float(val) == 1):
                    x_test[name] = float(val) / 10
                    break
                else:
                    print("The value you enter is out of range or the value is illegal character, please enter again: ")

        else:
            print("Something wrong with the attribute's title!")

    print(x_test)
    return x_test


if __name__ == "__main__":
    x_test = processInput()
    missing_list = is_empty(x_test)
    if len(missing_list) == 0:
        votingModels = VotingModels(x_test, 0.65)
        votingModels.voting()
    elif 0 < len(missing_list) < len(x_test):
        print("The prediction is \033[1;31;1mincomplete\033[0m! Because some attributes' value are missing:")
        print(missing_list)
        votingModels = VotingModels_2(x_test, 0.65)
        features_name = votingModels.extract_keys(x_test)
        model_search_result = votingModels.process_subfolder_names('../models', features_name)
        if model_search_result:
            print("We find most relevant models to do the prediction, here are results:")
            votingModels.voting()
        else:
            print("There is currently no relevant model available for predicting stroke.")
    else:
        print("All features are missing, there is no way to do the prediction.")
