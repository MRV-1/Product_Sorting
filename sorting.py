###################################################
# Sorting Products
###################################################

###################################################
# Application : Course Sorting
###################################################
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv(r"datasets\product_sorting.csv")
print(df.shape)
df.head(10)

# The comment count here is the sum of the 5,4,3,2,1 points of the dataset. But the data has been tampered with a bit and the fractions have been removed, so it may not be the sum when you drill down
# The aim here is to examine what problems are encountered when trying to rank these products.
# As a result of these examinations, we will develop a special ranking approach for ourselves and we will handle the scientific form of this with the statistical method.

####################
# Sorting by Rating
####################


df.sort_values("rating", ascending=False).head(20)

####################
# Sorting by Comment Count or Purchase Count
####################

# When ranked by number of purchases, I may not want to put bad courses up even if the number of purchases is high, I may want the user to somehow go for courses that they will be satisfied with.
# Getting the rating and reviews right is an important topic, but it can be a business decision to highlight products that might be good in some way when ranked with user satisfaction in mind.

df.sort_values("purchase_count", ascending=False).head(20)
df.sort_values("commment_count", ascending=False).head(20)

####################
# Sorting by Rating, Comment and Purchase
####################
# We aim to standardize all three metrics and be able to sort at the same time.

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.describe().T

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])


# The weights of each variable were created differently

(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)


def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

#bu tabloda elde edilen değerler güvenilirdir, 3 farklı yapı ağırlıklandırılarak bu sonuç elde edildi

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)
#elimizde sıralamayı etkileyebilecek birden fazla faktör olduğunda değişkenler aynı aralığa getirilmelidir, ardından istenirse ağırlıklandırılarak istenirse eşit ağırlık ile sıralama yapılır


####################
# Bayesian Average Rating Score
####################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

#rating'leri daha farklı açılardan ele alarak ya da sadece rating odaklı bir sıralama yapılabilir mi?

#bayesian_average_rating puan dağılımlarınının üzerinden ağırlıklı bir şekilde olasılıksal ortalama hesabı yapar.
#bu yöntem her bir kurs için ayrı ayrı olan puanların dağılım bilgisini kullanarak bir average rating hesabı yapmaktadır
#daha sonra buna göre sıralanabilir (bu durumda da bazı problemler olma ihtimali vardır) ya da hibrit bir yaklaşım geliştirilebilir
#bayesian_average_rating olasılıksal bir rating hesabıdır

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df.head()

#fonksiyona verilen n ifadesi girilecek olan yıldızların ve bu yıldızlara ait gözlenme frekanslarını ifade etmektedir
#diyelimki n 5 elemanlı birinci elemanında 1 yıldızdan kaç tane var, ikinci elemanında iki yıldızdan kaç tane var 5. elemanında 5 yıldızdan kaç tane var bilgileri girildikten sonra hesaplama işlemini gerçekleştirecektir.
#df'te görülen yıldız ifadeleri n ifadesinin yerine giriliyor olacak

#df'te görülen 5_point, 4_point, 3_point, 2_point, 1_point ifadeleri için öyle birşey yapılmalı ki tamamı seçilerek n ifadesine gönderiliyor olsun (ama ters sırada)
#confidence hesaplanacak olan z tablo değerine ilişkin bir değer elde edebilmek adına girilmiş bir değerdir, ön tanımlı olarak kabul edilebilir ya da burada da girilebilir
#bar_score ratinge odaklanarak bir sıralama sağlamıştır, rate'lerin dağılımına bakarak oluşturulmuş olan bu skor, rating hesabı içinde kullanılabilecek bilimsel çok değerli bir skordur
#bu durumda social prooflar yine gözden kaçtı(yorum sayısı, satın alma sayısı)
#ne olursa olsun tek odağımız verilen puanlar idi ise ve bu puanlara göre bir sıralama yapmak istiyor olacak isek bu durumda bar_score yöntemi tek başına kullanılabilir
#ama birden fazla  göz önünde bulundurulması gereken faktörler var ise bu durumda bar_score geçerli olmayacaktır


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

#NOT : df'e apply yapılırsa tüm değişken isimlerine kendi ismiyle ulaşabiliriz, df içerisine sütun ismi girildiği durumda x[0], x[1] şeklinde erişilebilir


df.sort_values("weighted_sorting_score", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)
#rating'e göre başlangıçta yapılan sıralamaya göre bu yapılan sıralamada sanki bazı kurslar çok yukarılara çıktı bunun sebebi bu yöntemde puanların dağılımına odaklanmış olması


df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)
#2. satırda yer alan kursun 1. satırda yer alan kurstan satın alma sayısı neredeyse 10 katı kadar fazla, daha fazla yorum sayısına da sahip, sebebi puan dağılımı
#çok daha az sayıda puana sahip olduğu halde, düşük puan miktarları diğer kursa göre daha az sayıda olmasından dolayı bu kurs yukarı çıkmıştır
#anlaşılıyorki bu yöntem rating'lerin dağılımına baktığından dolayı, sadece puana odaklandığından dolayı daha yüksek puanlara sahip olan dağılım açısından kurslar için hesapladığı puan çok daha yüksek

#burada da şöyle bir durum doğuyor, daha fazla satın alma ve yorum sayısına sahip ürün daha üst sıralara çıkmalıydı, başta konuşulan durum burada sağlanmamış oldu


####################
# Hybrid Sorting: BAR Score + Diğer Faktorler
####################

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Diğer Faktorler

#Bayesian Average Score puanları kırpmaktadır, bu olasılık yönteminden türeyen bir değerdir, olasılıksal bir ortalamadır
#Dolayısıyla bu değer kullanıldığında ilgili müşterilerinizin(kurumsal müşteriler)  Bayessian yöntemine göre bir rating hesabı yapıldığında bu durum kursların ya da ürünlerin olan puanlarını bir miktar kırpıp daha aşağıda göstereceğinden dolayı kullanılıp kullanılmaması tartışmaya açıktır ama tercihte edilebilir
#Bu ele alınan yöntemler bir araya getirilerek ağırlık olarak eklenebilir.

#wss kendi hesapladığımız yöntemdi
def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)
#bu çözüm hem iş bilgisi, hem bilimsellik hemde yeni potansiyel yıldızlara da şans veren bir şekilde sıralama işlemi gerçekleştirildi
#çıkan sonucu yorumlayacak olursak -->
#önceden satın almayı ve yorum sayısını göz önünde bulundurarak sıralama yapmak istiyordum ama diğer yandan ratinglerinde dikkate alınmasını bekliyordum,
#bu rating'lerle ilgili bir skor yöntemi daha gördüm ve bunlar harmanlandı

df.sort_values("hybrid_sorting_score", ascending=False).head(20)
#tablodan elde edileen sonuç 2. satırda bulunan kurs weighted_sorting_score'a göre daha yukarıda olmalıdır ama bayesian_average_rating'e göre 2.sırada yer almıştır

#buradaki sıralamayı wss score'a göre de yapmıştık buradan ne kazandık?
#Course_9'un üst sıralarda olması oldukça değerlidir kayda değer bir yorumu ve kayda değer bir satın alması var
#Course_1'in bulunduğu yer de önemli, yeni bir kurs yorum sayısı ve satın alma sayısı oldukça düşük ama puanı yüksek, bu potansiyel vaadediyor olabilir
#Bar_score bize yeni olsada potansiyel vaad edenleri yukarıya taşıma şansı sağlar,

#sıralama konusu dışardan bakıldığında çok basit gibi görünen içine girildiğinde çok fazla bussiness parametresi barındıran bir iştir
#bir kişi bir platformda ya da amazonda bir pazara girmek istiyor diyelim, pazara girmesi belirli bir zaman sonra teorik olarak imkansıza yakındır çünkü pazar dominantlığı vardır.
#bir şekilde yeni bir şeyler yapmak isteyenlerin bu alanlara girmesi çok zordur

#özetle bar score yöntemi hibrit bir sıralamada ağırlığı olan bir faktör olarak göz önünde bulundurulduğunda bir şekilde potansiyeli daha yüksek ama henüz yeterli social proof'u alamamış ürünleride yukarı çıkarmaktadır.
#5,4,3,2,1 dağılımları üzerinden olasılıksal olarak elde ettiğimiz score ise aslında potansiyeli ifade etmektedir

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)


############################################
# Uygulama: IMDB Movie Scoring & Sorting
############################################

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("C:/Users/MerveATASOY/Desktop/data_scientist-miuul/eğitim_teorik_içerikler/Bölüm_5_Measurement_Problems/vahit_hoca_projeler/measurement_problems/datasets/movies_metadata.csv",
                 low_memory=False)  # DtypeWarning kapamak icin

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape

#AMAÇ : Birçok insanın film izlemek için referans aldığı bu listeyi güncelle.
#İlk 250'yi sıralama yapma hedefiyle oluşturmamız beklenmektedir.

########################
# Vote Average'a Göre Sıralama
########################

df.sort_values("vote_average", ascending=False).head(20)

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T
#nereden filtre koymalıyım sorusuna yanıt arıyorum, filmlerin ortalama oy sayısı 100 civarındadır, bunun üzerinde bir değerlerde tutmalıyım mesela 400 alınabilir


#oy sayısı 400den büyük olanlara göre filtrele ve ardından sırala
df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)
#sadece vote_count'e göre bir filtreleme yaptım ve ardından sıraladım, vote_count'u 1 ile 10 arasına çektikten sonra vote_average ile çarpıp  sıralamak daha mantıklı olabilecektir.


from sklearn.preprocessing import MinMaxScaler
#bu işlemle yapılmak istenen şey buradaki vote_count değişkenini standartlaştırmaktır
df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])



########################
# vote_average * vote_count
########################

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score", ascending=False).head(20)


########################
# IMDB Weighted Rating
########################
#imdb çalışanlarının aldıkları karar neticesinde kullandıkları yöntem budur
#göz önünde bulunduracağım iki durum var 1--> kitlenin genel ortalaması 2--> sıralamaya girebilmek için gerekli  minimum oy sayısı olan m değeri
# v: ilgili filmin oy sayısı, m : gereken minimum oy sayısı, R : filmin puanı, c : genel bütün kitlenin ortalaması
# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)
#bir film eğer gerekenden fazla sayıda oy aldıysa bu filmin puanına uygulanacak olan düzeltmenin şiddeti az olacaktır, ama eğer film gerekenden daha az sayıda oy aldıysa asla bu listeye giremeyecektir, çünkü çok şiddetli bir düzeltmeye maruz kalacaktır



# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000     #1. filme göree aldığı oy sayısı fazla
               #demekki bir film eğer daha yüksek sayıda oy aldıysa bu durumda bu etki matematiksel bir şekilde tutulmuş oldu

# (3000 / (3000+500))*8 = 6.85

# (1000 / (1000+500))*9.5

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85

#aynı sayıda puan almış başka bir filmin puanı 9.5 olsun (1000 / (1000 + 500)) * 9.5 = 6.33 gelir
#vote average'i 8 olan filmin puanından kayda değer bir şekilde yüksek olduğu halde kendisinden daha fazla oy alan filmin önüne geçemedi,
#puan az olsa dahi daha fazla oya sahip olma durumu hepimizin tercih sebebi idi


# sağ kısımda az önce yapılan işlemin dengeleyicisi gibi bir işlem daha yapılmış

#ikinci filmin nihai sonucu daha yüksek çıktı

M = 2500
C = df['vote_average'].mean()

#formül fonksiyona taşındı
def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

#her bir film için bu hesaplama yapılmalı
df.sort_values("average_count_score", ascending=False).head(10)

weighted_rating(7.40000, 11444.00000, M, C)

weighted_rating(8.10000, 14075.00000, M, C)

weighted_rating(8.50000, 8358.00000, M, C)


#bütün veriye uygulayalım
df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(10)

#kendi iş probleminize göre ilgili konuyu matematiksel formda tanımlayabiliyorsanız bu durumda bu kullanılabilirdir


####################
# Bayesian Average Rating Score
####################

# 12481                                    The Dark Knight
# 314                             The Shawshank Redemption
# 2843                                          Fight Club
# 15480                                          Inception
# 292                                         Pulp Fiction

#imdb bu sıralama yöntemini değiştirdi, 2012 yılları civarında gözlemlediğimiz bayesian_average_rating yöntemini kullandı
#diğer faktörde user quality faktörleri, varsayıma göre muhtemelen imdb 2 faktöre göre bunu yaptı user quality ve bayesian yöntemi kullanarak yeni bir sıralama oluşturmuş olabilir




def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

#esaretin bedeli filmi için fonksiyona puanların dağılımları manuel gönderildi
bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

#baba filmi, kaçtane 1 yıldız almıi  kaçtane 2 yıldız almış, kaç tane 3 yıldız almış liste değerleri
bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])


#elimizdeki veri setinde puan sayısı ve averageler var ama bu puanların dağılımı yoktu yukarıdaki gibi el ile gönderileceğine aşağıdaki csv dosyasında bunlar verildi
df = pd.read_csv("C:/Users/MerveATASOY/Desktop/data_scientist-miuul/eğitim_teorik_içerikler/Bölüm_5_Measurement_Problems/vahit_hoca_projeler/measurement_problems/datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(20)

#e-ticaret dünyasında bir şekilde birden fazla ürünün kendi arasında yarışıyor olduğu senaryoların hemen hemen hepsinde en ciddi problem sıralama ve skorlama problemidir
#bu sıralama ve skorlama problemlerine birlikte bir bakış açısı geliştirildi
#user quality ağırlıkları var imdb'nin kullandığı biz onu baz alamadan hesapladık bu puanları
#zamana göre bazı trend yakalama çabalarıda sıralamada dikkate alınmalı

#BASİT BİR RECOMMENDER GELİŞTİRİLDİ ##

#IMDB sıralama hakkındaki bilgiler
# Weighted Average Ratings
# IMDb publishes weighted vote averages rather than raw data averages.
# The simplest way to explain it is that although we accept and consider all votes received by users,
# not all votes have the same impact (or ‘weight’) on the final rating.

# When unusual voting activity is detected,
# an alternate weighting calculation may be applied in order to preserve the reliability of our system.
# To ensure that our rating mechanism remains effective,
# we do not disclose the exact method used to generate the rating.
#
# See also the complete FAQ for IMDb ratings.
