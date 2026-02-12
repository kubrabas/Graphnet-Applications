# Graphnet-Applications

## ToDo:
1. Check last analysis calls
2. prepare resuls_analysis.ipynb. 
3. metrics.csv should include W at every epoches? plot train-val loss error. early stopping should monitor W?
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

