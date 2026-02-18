# Graphnet-Applications

## ToDo:
1. Check last analysis calls
2. 
3.  early stopping should monitor W?
4. check the log file, adjust your script if needed
5. understand everything in your combined script.
6. sinav kayit, tez kayit vs sorun var mi
7. graphnet source code'lardan daha cok helper kullan
8. scripti nasil hizlandirabilirim? gpu kullaniyor muyum? daha dogru nasil kullanabilirim? parallelization falan-
9. pone_offline ve graphnet_env pull et 
10. pone response scriptini de al Graphnet_applications icine
11. reindexed mevzusu dogru mu
12. yeni run icin degistirmem gereken path vs icin bir checklist'i bitir, gelistir.
13. train sbatch icine sey yaz, png olusturan bir script calissin. hatta mumkunse ipynb seklinde olabilir.
14. proje file'lari icinde source code'lari bulundurdugun dosyayi gelistir.
15. callbackler genel olarak ne? ogren.
16. ben sunu gordummmm out dosyasinda:
add_norm_layer: False
skip_readout: False
bunlari belki farkli kullanmak istersin küb.
17. micro-batch + gradient accumulation yontemi yapmisim ben. bu yontem ile effective olarak 128*8 = 1024 batch size condition elde ediyor muyum?  bunu belirleyen seylerden biri, layer normalization yapip yapmamam, dropout vs yapmamam sanirim. baska seyler de var. bunlari check et source koddan ve bendeki effective batch size ne tespit et. graphnet icin git pull yap. warningler vardi ya bir seyler yakinda bir yerde kullanilmicak diye. onlari coz.
18. data olusturma konusunda biraz daha calis. pom_response konusunda yani.
19. pmt_number kullanabilirsin. su an normal pmt posn kullaniyorsun.
20. mimariyi ayni makaledeki gibi yap. sonra farkli seyler dene.
21. birazcik cluster'i calis. ne kadar run edebilirim neyde edebilirim onda neler var vs.
22. noisy data denemelisin.
23. platolardan kurtulmak icin vsvs neler degistirip denemelisin? calis bunlari. ama once makalenin aynisini yapmayi ogren.
24. As a target variable, which one is better to use as unit: Zenith or radians? Maybe should try both?
25. hem zenith hem de azimuth makale ile uyumlu mu?
26. global parameters dogru ayarlanmis mi? homophily falan
27.  graphnet reponu duzenle, temizle, eski seyleri sil. 
28.  sorcagin sorulari biriktir
29. geometry optimization calis.
30. script basinda ve sonunda zamani print edelim. ya da direkt kalan zamani print edelim en son.
31. sunumun icin slayt hazirla. datani anlat. pom_response nasil olusturdun onu anlat. sonuclari goster.
32. loglarda hiz, sure vs de olsun. cpu gpu kullanimi vsvs. plot da olsun? ne kadar datada ne kadar computational source.
33. time_and_resources'i duzenle
34. zenith veya azimuth. biri yanlis bunlarin. yok ya degil sanirim.
35. genel log outputlarini kiyasla silebilecegin training orneklerini sil. 
36. .out dosyan daha organize, sectionlastirilmis olsun. daha 
37. data loaderlari ayirsam mi?
38. bu uzun training bitince tum output dosyalari vsvs chatgpt'ye ver ve genel yorum iste. bunu ayri ayri sohbetlerde yap. her dedigine odaklanip anlamaya calis. her dedgini ciddiye alip anlamaya calis.
39. [Test=zenith] batch 50/162 | median_abs_error_deg=4.462
gordum loglarda. ama energy icin bu sayi 50 degil 20 idi. neden?
40. zenith ve azimuth source kodlarini kiyasla.
41. batch size'i daha da arttirmaya calis. mumkun mu bu acaba. birden fazla gpu bi anda kullanabilir misin?
42. not: script 7 kendi rizamla durdurmusum. neden bilmiyom.
43. cmt sabah ilk is eski initiative scriptlerini sil (loglari inceleyerek). sonra son dokunuslari yap. push et. pazar gunu ve pazartesi gunu de geo optimization bakarsin.
44. hafta ici hocaya yaz. hocam sunumumsu bir sey hazirlamak istiyorum. ilk once bizim ozel meetingde sunayim diyorum. ne dersiniz de. sonra sizin feedbacklere gore ecp meeeting, pone analysis call'da sunarim de.
45. script/model inference sırasında log10(true_energy)’yi hesaplıyor, residual’ı üretiyor, ama CSV’ye ayrı bir kolon olarak yazmiyor. yazsin. sirasi da guzel olsun columnlarin
46. datani incele. training datan nasil ayriliyor test datan nasil ayriliyor.
47. toplam W ve diger ipynblerdeki plotlardan da ciz.
48. acaba AzimuthReconstruction mi kullanmaliyim AzimuthReconstructionWithKappa yerine
49. su ikisi de ornek.pdf’deki yaklaşımla uyumlu:
* `ZenithReconstructionWithKappa` ✅
* `AzimuthReconstructionWithKappa` ✅
ama tekrar check et. gpt dedi bunu.
50. hocaniy yolladigi paperlara bak.
51. hoca noiseless data ile devam et dedi. simdilik. simdilik bu sadece muon da ok.
52. hocayla konusmalarindan notlari yaz buraya
53. analysis call'de hangi source'lari kullandigini ve neler yaptigini outputlarla anlat kendini tanit.
54. gruba yaz hangi gun yapcaz de sor fln. design meetinge. 
55. elenaya da yaz ve sunumunu yap neler yaptigina dair.
56. caleb showers uretcek mi sorabilirsin. icecube ile de kiyasla, 10-10-10 ekleyerek de dene.
57. su anda su anki datanla devamkeeee.
58. icecube gen2 ile kiyasla. kendi yaptiklarini glb. 
59. showers icin Caleb'e sorabilirsin.
60. scriptte neden abs error degree? neden abs yani. degis onu mumkunse.
61. analysis scriptini incele. doru mu yapmissin zenith azimuth vs
62. VonMisesFisher2DLoss, radian bekliyor target'i. azimuth ve zenith siniflarin da oyleeee.
gpt said:
Evet. Bu üçüyle birlikte kullanacaksan, target açıların (azimuth ve zenith) radyan olmalı.

Neden?

VonMisesFisher2DLoss: sin/cos kullanıyor → PyTorch bunları radyan kabul eder.

AzimuthReconstructionWithKappa: atan2 ile açı üretiyor ve + 2π ile [0, 2π] aralığına getiriyor → radyan.

ZenithReconstructionWithKappa: sigmoid * π ile [0, π] aralığına map ediyor → radyan.

Dolayısıyla target’ların beklenen aralığı:

zenith ∈ [0, π]

azimuth ∈ [0, 2π]

Eğer target’ların dereceyse, eğitimden önce dönüştür.

63. zattirizortzort. PMT response uretirken lepton weighter'i da ekleee eehehehehehehe. sonra ona yonelik bir feature extractor vsvsvsvs