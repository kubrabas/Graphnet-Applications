# PMT response algorithm

Bu not sadece fiziksel / detector-response anlamina odaklanir.

Emin olmadigim yerleri en altta ayirdim. Burada yazilanlar P-ONE offline v2
modullerinde ve senin `apply_pmt_response_with_first_3_layers.py` tray
konfigurasyonunda gordugum davranisa dayaniyor.

## Fiziksel akis

Bir DAQ frame icin genel zincir:

1. CLSim'den gelen `I3Photons` fotonlari P-OM/PMT acceptance modelinden geciyor.
2. Kabul edilen fotonlar per-PMT photoelectron adaylarina (`I3MCPE`) cevriliyor.
3. Accepted signal hit'lerine dark noise ve K40 noise ekleniyor.
4. Signal + noise uzerine PMT zaman, charge, dead-time, late-pulse ve afterpulse response'u uygulanip `PMT_Response` uretiliyor.
5. Signal-only versiyona ayni PMT response uygulanip `PMT_Response_nonoise` uretiliyor.
6. Trigger noisy response uzerinden kuruluyor.
7. Trigger gecen frame'lerde 10000 ns uzunlugunda, trigger zamani 2000 ns'e resetlenmis `EventPulseSeries` yaziliyor.

## Acceptance modeli

Acceptance modelinin fiziksel anlami: CLSim fotonunun P-OM icindeki hangi PMT'ye
denk geldigini ve o fotonun photoelectron olarak algilanip algilanmayacagini
modellemek.

P-OM modeli:

- 16 PMT var.
- Modul radius: `0.2159 m`.
- PMT radius: `0.055 m`.
- Her PMT'nin yonu hard-coded PMT acilariyla tanimlaniyor.

Her foton icin:

1. Foton pozisyonu P-OM coordinate sistemine getirmek icin x ekseni etrafinda 90 derece rotate ediliyor.
2. Geometrik olarak hangi PMT'ye dustugu bulunuyor.
3. Foton sadece tam olarak bir PMT aperture'una dusuyorsa devam ediyor.
4. Hic PMT'ye dusmezse reddediliyor.
5. Birden fazla PMT'ye dusmesi kodda hata olarak ele aliniyor.

Detection probability:

```text
P_detect =
    photon.weight
  * quantum_efficiency(wavelength)
  * angular_acceptance(hit_distance)
  * glass_transmittance(wavelength)
  * collection_efficiency
```

Burada `collection_efficiency = 0.9`.

Sonra stochastic acceptance yapiliyor:

```text
random_uniform(0, 1) <= P_detect
```

ise foton kabul ediliyor. Kabul edilen her foton, ilgili PMT icin bir
photoelectron adayi gibi yaziliyor:

```text
time = photon.time
npe  = 1
```

Acceptance tablolarinin fiziksel icerigi su dosyalardan geliyor:

- `qe.csv`: wavelength'e bagli quantum efficiency
- `aa-0.csv`: angular acceptance
- `glass.csv`: wavelength'e bagli glass transmission

## Noise modeli

Noise, accepted signal zaman araligi etrafina ekleniyor. Manual bounds
kullanilmadigi icin pencere:

```text
ilk accepted hit zamani - 2000 ns
son accepted hit zamani + 10000 ns
```

### Dark noise

Dark noise PMT basina bagimsiz Poisson sureci gibi uretiliyor.

- Rate: `0.000001 pulses/ns`.
- Her PMT icin exponential inter-arrival time sample ediliyor.
- Her dark hit bir `I3MCPE` olarak `npe = 1` ile yaziliyor.

Fiziksel anlam: termal / elektronik dark count benzeri, signal'dan bagimsiz
tek-PMT noise hit'leri.

### K40 noise

K40 noise, karakterizasyon dosyasindan gelen rate, fold fraction ve PMT
kombinasyon dagilimlariyla uretiliyor.

Fiziksel anlam: potasyum-40 decay kaynakli, bazen tek PMT'ye bazen ayni
modulde birden fazla PMT'ye yakin-zamanli hit ureten correlated background.

Kod davranisi:

- K40 event time'lari exponential process ile uretiliyor.
- Event single-fold veya multi-fold olarak seciliyor.
- Multi-fold eventlerde PMT kombinasyonlari karakterizasyon histogramindan sample ediliyor.
- PMT kombinasyonu modulle simetrik dagilsin diye random flip ve 0/90/180/270 derece rotation uygulanabiliyor.
- Arrival-time spread karakterizasyonundan zaman offsetleri sample ediliyor.

## PMT response modeli

`DOMSimulation`, accepted signal ile dark + K40 noise'u birlestirip PMT pulse
response'a ceviriyor.

Kullanilan ana parametreler:

- Transit-time spread: `3 ns`
- PMT transit time: `25 ns`
- Charge mean: `1.0`
- Charge sigma: `0.3`
- Afterpulse probability: `0.06`
- Early afterpulse mean/sigma: `2000 ns / 1000 ns`
- Late afterpulse mean/sigma: `8000 ns / 2000 ns`
- Early afterpulse component ratio: `0.3`
- Late-pulse probability: `0.01`
- PMT dead time: `10 ns`
- PE threshold: `0.25`
- PE saturation: `100.0`
- Pulse merge separation in this worker: `0.2 ns`

Signal hit'leri icin:

- Zaman transit-time spread ile smear ediliyor.
- `1%` olasilikla late pulse ekleniyor.
- `6%` olasilikla afterpulse ekleniyor.

Dark/K40 noise hit'leri icin:

- Zaman transit-time spread ile smear ediliyor.
- Afterpulse uygulanabiliyor.
- Late pulse kapali gorunuyor (`late_pulses=False`).

Sonra signal-only map ve signal+noise map ayri isleniyor:

- `PMT_Response_nonoise`: sadece signal'dan gelen pulse'lar.
- `PMT_Response`: signal + dark + K40 pulse'lari.
- `triggerpulsemap`: trigger icin kullanilan signal + dark + K40 pulse'lari.

Pulse olusumu sirasinda:

1. Ayni PMT'de 10 ns dead time icindeki ardil hit'ler atiliyor.
2. Her kalan hit icin charge Gaussian sample ediliyor: mean `1.0`, sigma `0.3`.
3. Cok yakin pulse'lar birlestirilebiliyor.
4. Charge `< 0.25` ise pulse atiliyor.
5. Charge buyukse saturation modeli uygulanarak charge sinirlaniyor.

## Trigger kosulu

Bu kisim fiziksel olarak en kritik nokta.

Frame'in yazilabilmesi icin iki kosul var:

```text
1. PMT_Response icinde en az 5 farkli OM hit olmali.
2. Noisy response uzerinden en az bir DOM 3-PMT coincidence trigger vermeli.
```

Birinci kosul `HitCountCheck(NHits=5)`:

```text
unique (string, om) count >= 5
```

Ikinci kosul DOM-level trigger:

```text
ayni DOM icinde
10 ns pencere icinde
en az 3 farkli PMT hit'i
```

Bu trigger noisy map uzerinden hesaplandigi icin dark noise ve K40 noise
trigger'a katkida bulunabilir.

`DetectorTrigger` konfigurasyonunda isim `_3PMT_1DOM`; burada
`FullDetectorCoincidenceN=1` verilmis. Ancak bu konfigurasyonda detector/string
coincidence branch'i pratikte devre disi kaliyor, cunku default
`SingleOMTriggerCoince=3` ve `OMPMTCoinc=3` oldugunda kod `DoCoincTriggers=False`
yapiyor. Bu nedenle efektif trigger, single-DOM 3-PMT trigger olarak calisiyor.

Kisa hali:

```text
writer'a giden DAQ frame =
    >= 5 unique OM
AND >= 1 DOM with >= 3 distinct PMTs inside 10 ns
```

## EventPulseSeries nasil uretiliyor?

Trigger gecen frame'de en erken trigger zamani bulunuyor. Bu konfigurasyonda
ilgili trigger zamani single-DOM trigger zamanindan geliyor.

Sonra `PMT_Response` pulse'lari su pencereye gore seciliyor:

```text
min_trigger_time - 2000 ns < pulse.time < min_trigger_time + 8000 ns
```

Secilen pulse'larin zamani resetleniyor:

```text
new_time = pulse.time - min_trigger_time + 2000 ns
```

Yani event penceresi:

- toplam uzunluk: `10000 ns`
- trigger'in yerlestirildigi zaman: `2000 ns`
- trigger oncesi pencere: `2000 ns`
- trigger sonrasi pencere: `8000 ns`

`EventPulseSeries` noisy pulse'lardan uretiliyor.

`EventPulseSeries_nonoise` ise `PMT_Response_nonoise` pulse'larindan uretiliyor,
ama frame zaten daha once noisy trigger ile hayatta kalmis oluyor. Yani
`EventPulseSeries_nonoise` kendi basina ayri bir no-noise trigger kosulu koymuyor.

## Fiziksel yorum

Bu algoritma, photon propagation sonucu gelen ideal photon hit'lerini daha
detector-like pulse'lara ceviriyor:

- geometrik PMT acceptance,
- wavelength-dependent QE,
- angular acceptance,
- glass transmission,
- collection efficiency,
- stochastic detection,
- dark noise,
- correlated K40 noise,
- PMT timing smear,
- late pulse,
- afterpulse,
- dead time,
- charge response,
- threshold/saturation,
- 3-PMT-in-1-DOM local trigger.

Bu nedenle final `EventPulseSeries`, saf photon simulation degil; trigger ve
readout penceresi uygulanmis, noise dahil detector response seviyesindeki pulse
serisi.

## Emin olmadigim / yorum yapmadigim yerler

- `qe.csv`, `aa-0.csv`, `glass.csv` ve `k40-characterization.hdf5` dosyalarinin
  deneysel / kalibrasyon kokenini dogrulamadim. Sadece algoritmada nasil
  kullanildiklarini dogruladim.
- `I3MCPE.npe` alaninin `DOMSimulation` icinde gecici olarak PMT numarasi
  tasimasi fiziksel bir `npe` yorumu degil; bu sadece kod davranisi.
- Detector/string coincidence branch'inin kapali olmasinin fiziksel olarak
  kasitli olup olmadigini bilmiyorum. Sadece bu konfigurasyonda efektif trigger
  single-DOM 3-PMT olarak calisiyor.
- Bu not full IceTray job calistirilarak degil, source-code inspection ile
  yazildi.

## Kod kaynaklari

Fiziksel model icin baktigim kaynaklar:

- `DataPreperation/PmtResponse/apply_pmt_response_with_first_3_layers.py`
- `/cvmfs/software.pacific-neutrino.org/pone_offline/v2.0/DOM/OMAcceptance.py`
- `/cvmfs/software.pacific-neutrino.org/pone_offline/v2.0/Utilities/POMModel.py`
- `/cvmfs/software.pacific-neutrino.org/pone_offline/v2.0/NoiseGenerators/DarkNoise.py`
- `/cvmfs/software.pacific-neutrino.org/pone_offline/v2.0/NoiseGenerators/K40Noise.py`
- `/cvmfs/software.pacific-neutrino.org/pone_offline/v2.0/DOM/PONEDOMLauncher.py`
- `/cvmfs/software.pacific-neutrino.org/pone_offline/v2.0/Trigger/DOMTrigger.py`
- `/cvmfs/software.pacific-neutrino.org/pone_offline/v2.0/Trigger/DetectorTrigger.py`
