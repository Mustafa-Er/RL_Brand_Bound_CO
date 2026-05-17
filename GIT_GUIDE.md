# Git Rehberi (Yeni Başlayanlar İçin)

Bu rehber 2 kişilik takım için en temel Git iş akışını anlatır. Komutlar
Ubuntu/WSL terminalinden çalıştırılır.

---

## 0. Tek seferlik kurulum

```bash
git config --global user.name "Adınız Soyadınız"
git config --global user.email "siz@ornek.com"
git config --global pull.rebase true   # pull yaparken merge commit değil rebase
```

GitHub'a kimlik doğrulama için **Personal Access Token (PAT)** kullanın
(Settings → Developer settings → Personal access tokens). İlk push'ta kullanıcı
adınız ve parola olarak token sorulur.

---

## 1. İlk push (proje sahibi)

Repo zaten `https://github.com/Mustafa-Er/RL_Brand_Bound_CO` adresinde boş
olarak duruyor. Yerelde:

```bash
cd RL_Brand_Bound_CO
git init -b main
git add .
git commit -m "chore: initial project skeleton"
git remote add origin https://github.com/Mustafa-Er/RL_Brand_Bound_CO.git
git push -u origin main
```

---

## 2. Takım arkadaşının repoyu alması

```bash
git clone https://github.com/Mustafa-Er/RL_Brand_Bound_CO.git
cd RL_Brand_Bound_CO
conda env create -f environment.yml
conda activate rl_bb
```

---

## 3. Günlük döngü

Çalışmaya başlamadan önce **her zaman** güncel ana dalı çekin:

```bash
git checkout main
git pull
```

### Feature branch açma

Yeni bir iş parçası için branch açın (kod, doğrudan `main`'e yazmayın):

```bash
git checkout -b feat/instance-generator
# ... kod yaz, dosya düzenle ...
git add src/rl_bb/instances/generate.py
git commit -m "feat: add set covering instance generator"
git push -u origin feat/instance-generator
```

GitHub'da Pull Request açın, takım arkadaşı `main`'e merge etmeden önce
gözden geçirsin.

### Conventional Commits örnekleri

- `feat: ...` → yeni özellik
- `fix: ...` → hata düzeltme
- `chore: ...` → yapı/araç değişiklikleri
- `docs: ...` → sadece doküman
- `refactor: ...` → davranışı değiştirmeyen düzenleme
- `test: ...` → test ekleme/güncelleme

---

## 4. Takım arkadaşının değişikliklerini almak

```bash
git checkout main
git pull
```

Kendi feature branch'inizdeyseniz ve `main`'deki güncellemeleri branch'inize
taşımak istiyorsanız:

```bash
git checkout feat/benim-branch
git pull --rebase origin main
```

---

## 5. Merge conflict (çakışma) çözme

İki kişi aynı dosyanın aynı satırlarını değiştirdiyse Git çakışma bildirir.
Örnek:

```bash
git pull --rebase origin main
# CONFLICT (content): Merge conflict in src/rl_bb/utils/__init__.py
```

1. Çakışan dosyayı açın. Şu bloğu görürsünüz:
   ```
   <<<<<<< HEAD
   sizin satırınız
   =======
   diğer kişinin satırı
   >>>>>>> main
   ```
2. İstediğiniz nihai içeriği elle yazın, `<<<<<<<`, `=======`, `>>>>>>>`
   satırlarını silin.
3. Dosyayı kaydedip:
   ```bash
   git add src/rl_bb/utils/__init__.py
   git rebase --continue
   ```
4. Sonra push edin (rebase yaptıysanız force-with-lease gerekebilir):
   ```bash
   git push --force-with-lease
   ```

> **İpucu:** `git status` her zaman size sonraki adımın ne olduğunu söyler.
> Kaybolduğunuzda ilk komut o.

---

## 6. Aynı dosyaya iki kişi commit ettiğinde

Aynı dosyayı **farklı yerlerinden** düzenlediyseniz Git otomatik birleştirir.
**Aynı yerinden** düzenlediyseniz Bölüm 5'teki çakışma akışı devreye girer.
İki kişi aynı feature üzerinde çalışacaksa, mümkünse aynı branch üzerinden
pull/push yapın veya işi dosya bazında bölüştürün.

---

## 7. Tehlikeli komutlar (acil olmadıkça kullanmayın)

- `git reset --hard` → yerel değişiklikleri tamamen siler
- `git push --force` → uzaktaki geçmişi ezer; **`--force-with-lease` tercih edin**
- `git clean -fd` → izlenmeyen dosyaları siler

Bu komutlardan birini çalıştırmadan önce takım arkadaşınıza haber verin.
