## ToDo
1. prepera resuls_analysis.ipynb: calculate W, plot train-val loss error.
2. ? # ## micro-batch + gradient accumulation yontemi yapmisim ben. bu yontem ile effective olarak 128*8 = 1024 batch size condition elde ediyor muyum?  bunu belirleyen seylerden biri, layer normalization yapip yapmamam, dropout vs yapmamam sanirim. baska seyler de var. bunlari check et source koddan ve bendeki effective batch size ne tespit et. graphnet icin git pull yap. warningler vardi ya bir seyler yakinda bir yerde kullanilmicak diye. onlari coz cnmmmmm.
3. scriptlerini bastan incele her sey ok mu. anlamadiigign seyleri anla
4. pone_offline ve graphnet_env pull et :) 
5. yeni run icin degistirmem gereken path vs icin bir checklist olustur
6. warningleri coz. graphnet version ile alakali olan.
7. tüm todolari bir yerde topla
8. reindexed mevzusu dogru mu
9. su tarz errorlari atlasin? 
[1;34mgraphnet[0m [SpawnProcess-9] [33mWARNING [0m 2026-02-02 06:37:06 - has_jammy_flows_package - `jammy_flows` not available. Normalizing Flow functionality is missing.[0m
[1;34mgraphnet[0m [SpawnProcess-9] [33mWARNING [0m 2026-02-02 06:37:06 - has_icecube_package - `icecube` not available. Some functionality may be missing.[0m
[1;34mgraphnet[0m [SpawnProcess-9] [33mWARNING [0m 2026-02-02 06:37:06 - has_km3net_package - `km3net` not available. Some functionality may be missing.[0m
[1;34mgraphnet[0m [SpawnProcess-9] [33mWARNING [0m 2026-02-02 06:37:06 - <module> - `graphnet.models.graphs` will be depricated soon. All functionality has been moved to `graphnet.models.data_representation`.[0m
[1;34mgraphnet[0m [SpawnProcess-10] [33mWARNING [0m 2026-02-02 06:37:12 - has_jammy_flows_package - `jammy_flows` not available. Normalizing Flow functionality is missing.[0m
[1;34mgraphnet[0m [SpawnProcess-10] [33mWARNING [0m 2026-02-02 06:37:12 - has_icecube_package - `icecube` not available. Some functionality may be missing.[0m
[1;34mgrap
10. metrics csv bozuk. duzelt
11. sinav kayit, tez kayit vs sorun var mi
12. ben sunu gordummmm out dosyasinda:
add_norm_layer: False
skip_readout: False
13. pone response scriptini de al Graphnet_applications icine
14. train sbatch icine sey yaz, png olusturan bir script calissin. hatta mumkunse ipynb seklinde olabilir.
15. ValOpeningAngleLogger.on_validation_batch_end içinde pl_module(batch) ile tekrar inference yapıyorsun
16. inference kisminda bir yerde cpu yaziodu neden
17.
#### source code'lari ver gpt'ye. sonra bu scripti ver. sonra sor. neleri degistirmeliyim de.
#### task head'i var ya, onu anla. degistirmen gereken bir sey varsa degistir. task makale ile de ayni mi ogren.
#### callbackler genel olarak ne? ogren.
#### bu scripti genel olarak anlamaya calis
# global epoch context 
## genel olarak scriptte her sey ok mu? source code'lari okuyarak karar ver.



## Training Initiative 1
##### Info: Energy Reco No Noise


## Training Initiative 2
##### Info: Energy Reco No Noise
1. out and err files are combined into one out file and the file is now located in the corresponding initiative folder
2.  max_epochs increased to 75,  early_stopping_patience increased to 15
3. out file cleaned a bit. now only epoch level info will be printed. not each batch in each epoches.
4. mem is decreased: "#SBATCH --mem=32G"
5. this error is solved: `graphnet.models.graphs` will be depricated soon. All functionality has been moved to `graphnet.models.data_representation`
6. metrics.csv is prepared to monitor/compare validation-training error curve
    
## Training Initiative 3
##### Info: Angle Reco No Noise
1. objective changed: now we train 3D direction derived from (azimuth, zenith). We add Direction() label (az+ze → direction vector). Task: DirectionReconstructionWithKappa, Loss: VonMisesFisher3DLoss (predicts direction + kappa uncertainty/confidence). Test outputs: opening_angle_deg, kappa, true/pred azimuth-zenith.
2. Improvement in Early Stopping logs: Epoch Number is included
3. Validation metric upgrade: opening angle quantiles + kappa mean
4. Sanity checks added (direction norm ≈ 1, az/ze range)
5. I haven’t verified yet whether this matches the reference paper’s exact setup. I will do this in the initiative4

    
## Training Initiative 4
##### Info: Angle Reco No Noise

1. **Imports cleaned:** removed duplicate imports and consolidated everything into a single import block.
2. **Removed redundant GraphNeT imports:** deleted the second GraphNeT import section (`E402/noqa` clutter removed).
3. **Logging/epoch wiring cleaned up:** moved log-filter setup into a guarded `install_logging_filters()` and call it from `run()`; **epoch context hardened:** `_EpochContextCallback` now updates epoch on both train and validation epoch start (so the existing `epoch: X` injection stays consistent).
4. just a test run. 10 epoch only
