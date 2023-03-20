###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
# 1. flo_data_20K.csv verisini okuyunuz.
# 2. Veri setinde
# a. İlk 10 gözlem,
# b. Değişken isimleri,
# c. Betimsel istatistik,
# d. Boş değer,
# e. Değişken tipleri, incelemesi yapınız.
# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
# 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.


import pandas as pd
import datetime as dt

df_ = pd.read_csv(
    r"\FLOMusteriSegmentasyonu\flo_data_20k.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = df_.copy()
df.head(10)
df["master_id"].nunique()
df.shape
df.describe().T
df.isna().sum()
df.info()
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df[["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]] = df[
    ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]].apply(
    lambda x: pd.to_datetime(x))
df.groupby("order_channel").agg({
    "master_id": "count",
    "order_num_total": "mean",
    "customer_value_total": "mean"
})
df.sort_values(by="customer_value_total", ascending=False).head(10)
df.sort_values(by="order_num_total", ascending=False).head(10)


def data_prep(df):
    df.head(10)
    df["master_id"].nunique()
    df.shape
    df.describe().T
    df.isna().sum()
    df.info()
    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    df[["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]] = df[
        ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]].apply(
        lambda x: pd.to_datetime(x))
    df.groupby("order_channel").agg({
        "master_id": "count",
        "order_num_total": "mean",
        "customer_value_total": "mean"
    })
    df.sort_values(by="customer_value_total", ascending=False).head(10)
    df.sort_values(by="order_num_total", ascending=False).head(10)
    return df


df

df = data_prep(df)
# GÖREV 2: RFM Metriklerinin Hesaplanması

df["last_order_date"].max()
today_date = df["last_order_date"].max() + dt.timedelta(days=2)

df["recency"] = (today_date - df["last_order_date"]).dt.days
df["frequency"] = df["order_num_total"]
df["monetary"] = df["customer_value_total"]

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması
df.head()
df = df.loc[(df["monetary"] > 0) & (df["frequency"] > 0)]

df["recency_score"] = pd.qcut(df["recency"], 5, labels=[5, 4, 3, 2, 1])
df["frequency_score"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
df["monetary_score"] = pd.qcut(df["monetary"], 5, labels=[1, 2, 3, 4, 5])

df["RFM_score"] = df["recency_score"].astype(str) + df["frequency_score"].astype(str)

df.head(10)
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

df['segment'] = df['RFM_score'].replace(seg_map, regex=True)
df.head(10)
# GÖREV 5: Aksiyon zamanı!
# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

df[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
# ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
# yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.

target_customers = df.loc[((df["segment"] == "champions") | (df["segment"] == "loyal_customers"))]
target_customers = target_customers[target_customers["interested_in_categories_12"].apply(lambda x: 'KADIN' in x)]
target_customers = target_customers[target_customers["monetary"] > 250]

target_customers["master_id"].to_csv("target_customer_ids.csv")

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
# alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.

target_customers_sec = df.loc[((df["segment"] == "about_to_sleep")
                               | (df["segment"] == "cant_loose")
                               | (df["segment"] == "hibernating")
                               | (df["segment"] == "new_customers"))]

target_customers_sec.head(10)

target_customers_sec = target_customers_sec[
    target_customers_sec["interested_in_categories_12"].apply(lambda x: 'COCUK' in x)
    | target_customers_sec["interested_in_categories_12"].apply(lambda x: 'ERKEK' in x)]

target_customers_sec.head(10)

target_customers_sec["master_id"].to_csv("target_customer_sec_ids.csv")

# GÖREV 6: Tüm süreci fonksiyonlaştırınız.


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


def create_rfm(dataframe, csv=False):
    df_new = dataframe
    df_new["last_order_date"].max()
    today_date_func = dt.datetime(2021, 6, 1)

    df_new["recency"] = (today_date_func - df_new["last_order_date"]).dt.days
    df_new["frequency"] = df_new["order_num_total"]
    df_new["monetary"] = df_new["customer_value_total"]
    df_new = df_new.loc[(df_new["monetary"] > 0) & (df_new["frequency"] > 0)]

    df_new["recency_score"] = pd.qcut(df_new["recency"], 5, labels=[5, 4, 3, 2, 1])
    df_new["frequency_score"] = pd.qcut(df_new["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    df_new["monetary_score"] = pd.qcut(df_new["monetary"], 5, labels=[1, 2, 3, 4, 5])

    df_new["RFM_score"] = df_new["recency_score"].astype(str) + df_new["frequency_score"].astype(str)

    df_new['segment'] = df_new['RFM_score'].replace(seg_map, regex=True)
    df_new[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
    target_customers_func = df_new.loc[((df_new["segment"] == "champions") | (df_new["segment"] == "loyal_customers"))]
    target_customers_func = target_customers[target_customers_func["interested_in_categories_12"].apply(lambda x: 'KADIN' in x)]
    target_customers_func = target_customers[target_customers_func["monetary"] > 250]
    target_customers_sec_func = df_new.loc[((df_new["segment"] == "about_to_sleep")
                                       | (df_new["segment"] == "cant_loose")
                                       | (df_new["segment"] == "hibernating")
                                       | (df_new["segment"] == "new_customers"))]

    target_customers_sec_func = target_customers_sec_func[
        target_customers_sec_func["interested_in_categories_12"].apply(lambda x: 'COCUK' in x)
        | target_customers_sec_func["interested_in_categories_12"].apply(lambda x: 'ERKEK' in x)]

    if csv:
        target_customers_func["master_id"].to_csv("target_customer_ids.csv")
        target_customers_sec_func["master_id"].to_csv("target_customer_sec_ids.csv")
    return df_new



df_2 = df.copy()

df_2 = create_rfm(df_2)
df_2 = create_rfm(df_2, csv=True)

