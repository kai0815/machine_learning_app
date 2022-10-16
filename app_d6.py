#ライブラリの読み込み
import time
from scipy.fft import irfft
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler #スケーリングの為に追記
import joblib # モデル保存用
from io import BytesIO

#stremlitがリロードされない用にcallback関数を定義
def callback():
      st.session_state.key=True
      

# 重回帰分析_学習の実行関数　を定義
@st.cache(allow_output_mutation=True)
def ex_model_linear(X_train, y_train):
      lr = linear_model.LinearRegression()
      lr.fit(X_train, y_train)

      return lr 

# 重回帰分析_スケーリングの実行関数　を定義
@st.cache(allow_output_mutation=True)
def ex_model_scaling(X_train , X_test):
      scaler=StandardScaler() #スケーリング宣言
      scaler.fit(X_train)  # scalerの学習　←平均と標準偏差を計算
      X_train2 = scaler.transform(X_train) #scaling
      X_test2 = scaler.transform(X_test)  #scaling     
      
      return X_train2 , X_test2 


# 重回帰分析_スケーリング_学習の実行関数　を定義
@st.cache(allow_output_mutation=True)
def ex_model_linear_scaling(X_train2, y_train):
      lr2 = linear_model.LinearRegression() #scalling用
      lr2.fit(X_train2 , y_train)     
      
      return lr2 



#タイトル
st.title("機械学習アプリ")


# 以下をサイドバーに表示
st.sidebar.markdown("### 機械学習に用いるcsvファイルを入力してください")
#ファイルアップロード
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files= False)
#ファイルがアップロードされたら以下が実行される
if uploaded_files:
    df = pd.read_csv(uploaded_files)
    df_columns = df.columns
    data_num=len(df)
    #データフレームを表示
    st.markdown("### 入力データ")
    st.write('データ数は',data_num)
    st.dataframe(df.style.highlight_max(axis=0))
    #統計量と相関係数を表示  
    st.markdown("### 統計量")
    st.write(df.describe())
    st.markdown("### 相関係数")
    st.write(df.corr())
    
    #matplotlibで可視化。X軸,Y軸を選択できる
    st.markdown("### 可視化 単変量")
    #データフレームのカラムを選択オプションに設定する
    x = st.selectbox("X軸", df_columns)
    y = st.selectbox("Y軸", df_columns)
    #選択した変数を用いてmtplotlibで可視化
    
    fig = plt.figure(figsize= (12,8))
    plt.scatter(df[x],df[y])
    plt.xlabel(x,fontsize=18)
    plt.ylabel(y,fontsize=18)
    st.pyplot(fig)

    #seabornのペアプロットで可視化。複数の変数を選択できる。
    st.markdown("### 可視化 ペアプロット")
    #データフレームのカラムを選択肢にする。複数選択
    item = st.multiselect("可視化するカラム", df_columns)
    #散布図の色分け基準を１つ選択する。カテゴリ変数を想定
    hue = st.selectbox("色の基準", df_columns)
    
    #実行ボタン（なくてもよいが、その場合、処理を進めるまでエラー画面が表示されてしまう）
    execute_pairplot = st.button("ペアプロット描画")
    #実行ボタンを押したら下記を表示
    if execute_pairplot:
            df_sns = df[item]
            df_sns["hue"] = df[hue]
            
            #streamlit上でseabornのペアプロットを表示させる
            fig = sns.pairplot(df_sns, hue="hue")
            st.pyplot(fig)


    st.markdown("### モデリング")
    #説明変数は複数選択式
    ex = st.multiselect("説明変数を選択してください（複数選択可）", df_columns)

    #目的変数は一つ
    ob = st.selectbox("目的変数を選択してください", df_columns)

    #機械学習のタイプを選択する。
    ml_menu = st.selectbox("実施する機械学習のタイプを選択してください", ["重回帰分析","ロジスティック回帰分析"])





      


    #機械学習のタイプにより以下の処理が分岐
    if "key" not in st.session_state: # 何らかの操作でリロードされない為に変数保持
      st.session_state.key=False
    if ml_menu == "重回帰分析" or st.session_state.key:
            st.markdown("#### 機械学習を実行します")
            testsize=st.number_input('テストサイズを入力',0.0,1.0,0.3)
            execute = st.button("実行", on_click=callback)
            
            
            
            
            
            #実行ボタンを押したら下記が進む
            
            if execute or st.session_state.key:
                  st.write(st.session_state.key)
                  df_ex = df[ex]
                  df_ob = df[ob]
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size = testsize , random_state=1)
                  lr = ex_model_linear(X_train,y_train)

                  #スケーリングパート追記
                  X_train2 , X_test2 = ex_model_scaling(X_train , X_test)
                  lr2 = ex_model_linear_scaling(X_train2 , y_train)
                  
                  #小数点表示へ変換
                  np.set_printoptions(precision=2,suppress=True)
                  # 重みの確認
                  st.markdown("### 説明変数の重み")
                  df_coef=pd.DataFrame(lr2.coef_,index=ex)
                  st.dataframe(df_coef)
                  
                  #プログレスバー（ここでは、やってる感だけ）
                  my_bar = st.progress(0)
                  
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)
                  
                  #metricsで指標を強調表示させる
                  col1, col2 = st.columns(2)
                  col1.metric(label="トレーニングスコア", value=lr.score(X_train, y_train))
                  col2.metric(label="テストスコア", value=lr.score(X_test, y_test))

                  #モデルの可視化
                  st.markdown("### モデルの可視化")
                  fig2 = plt.figure(figsize= (12,8))
                  for i in range(len(df)):
                        x=df_ex.iloc[i,:]
                        y_pred=lr.predict([x])
                        plt.plot(i,y_pred,"*",color="r")
                  plt.plot(df_ob,"o",label="data",color="b")
                  plt.xlabel("Number",fontsize=18)
                  plt.ylabel(ob,fontsize=18)
                  plt.legend()

                  #モデル予測値と生データの残差を可視化
                  st.pyplot(fig2)
                  st.markdown("### モデルの可視化 -モデル予測値と生データの残差-")
                  y_pred_all=pd.DataFrame() #空のデータフレームを用意
                  for i in range(len(df)):

                        x=df_ex.iloc[i,:]
                        y_pred=lr.predict([x])
                        y_pred_pd=pd.DataFrame(y_pred,index=[i]) #numpy配列→pandasへ変換
                        y_pred_all=y_pred_all.append(y_pred_pd)

                  y_diff = y_pred_all[0] - df_ob
                  y_diff_pd=pd.DataFrame(y_diff,columns=["diff"])

                  # 横軸：Number 縦軸：誤差のグラフ作成
                  fig3=plt.figure(figsize=(12,8))
                  plt.plot(y_diff,"o",color="r")
                  plt.xlabel("Number",fontsize=18)
                  plt.ylabel('diff(model - act)',fontsize=18)
                  st.pyplot(fig3)
                  checkbox = st.checkbox("モデル予測値と実値の差分_デジタル値の表示")
                  
                  if checkbox:
                        st.dataframe(y_diff_pd,width=200 , height=200)
                  
                  #↑の横軸を自由に選べるグラフ作成
                  ex2 = st.selectbox('x軸にする変数を選んでください',df_columns,on_change=callback)
                  fig4=plt.figure(figsize=(12,8))
                  plt.plot(df[ex2], y_diff,"o",color="r")
                  plt.xlabel(ex2,fontsize=18)
                  plt.ylabel('diff(model - act)',fontsize=18)
                  st.pyplot(fig4)

                  #モデルのダウンロード
                  model_io = BytesIO()
                  joblib.dump(lr , model_io)
                  st.download_button("Download model" , model_io.getvalue())

                  """ joblib.dump(lr , "model.pkl")
                  with open("model.pkl" , "rb") as model:
                        st.download_button("Download model" , model) """
                  
                  
                  



                  #予測値の出力




                  
                        
                  
    #ロジスティック回帰分析を選択した場合
    elif ml_menu == "ロジスティック回帰分析":
            st.markdown("#### 機械学習を実行します")
            testsize=st.number_input('テストサイズを入力',0.0,1.0,0.3)
            execute = st.button("実行")
            
            lr = LogisticRegression()

            #実行ボタンを押したら下記が進む
            if execute:
                  df_ex = df[ex]
                  df_ob = df[ob]
                  X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size = testsize)
                  lr.fit(X_train, y_train)
                  #プログレスバー（ここでは、やってる感だけ）
                  my_bar = st.progress(0)
                  for percent_complete in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent_complete + 1)

                  col1, col2 = st.columns(2)
                  col1.metric(label="トレーニングスコア", value=lr.score(X_train, y_train))
                  col2.metric(label="テストスコア", value=lr.score(X_test, y_test))