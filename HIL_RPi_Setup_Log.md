# HIL Testbed — RPi Hardware Setup Log

Raspberry Pi 5 + Arduino Uno R3 | June 2026, networking updated October 2026

---

## 1. SD Card & OS Flashing

### 1.1 Decision: Fresh Flash

The retail bundle SD card was discarded in favour of a clean flash. Reasons:

- pyserial serial communication is sensitive to OS-level USB/serial driver state
- pandapower 3.4.0 + simbench 1.6.1 have specific Python version dependencies
- juliacall requires a clean environment for proper isolation
- Retail bundle cards often ship with bloatware or outdated packages

### 1.2 OS Selection

Selected: Raspberry Pi OS Lite (64-bit), Bookworm. Lite chosen over Desktop — all project code runs via SSH, no GUI required on the RPi itself. The dashboard is served over the network to the laptop browser, so the RPi does not need a desktop. 64-bit is mandatory — RPi 5 requires aarch64.

### 1.3 Raspberry Pi Imager Settings

- Device: Raspberry Pi 5
- OS: Raspberry Pi OS Lite (64-bit) — listed under "Raspberry Pi OS (other)"
- Hostname: `HIL-RPi`
- Username: `ppre_hil`
- SSH: Enabled (password authentication)
- Keyboard layout: US (match physical laptop keyboard, not geographic location)
- Timezone: Europe/Berlin
- WiFi: Laptop Windows Mobile Hotspot SSID + password
- RPi Connect: Skipped — direct SSH used instead

### 1.4 Troubleshooting: SD Card Write Error

Error: *"Error writing to storage device. Please check if the device is writable, has sufficient space, and is not write-protected."*

Resolution — use Diskpart to fully clean the card:

```
Win + R -> diskpart -> Enter (run as admin)
list disk
select disk X        <- match by 32GB size
clean
create partition primary
format fs=fat32 quick
assign
exit
```

---

## 2. Network, eduroam & SSH Access

> **Revised October 2026.** This whole section was rewritten. The original procedure assumed `wpa_supplicant.conf` and a manual WPA2-Enterprise eduroam configuration. Both assumptions are wrong for this machine. See [2.9](#29-historical-note--superseded) for what the original log said and why it no longer applies.

### 2.1 Network stack: NetworkManager, not wpa_supplicant

Raspberry Pi OS Bookworm manages networking with NetworkManager. Editing `/etc/wpa_supplicant/wpa_supplicant.conf` has no effect on this machine and will fail silently — most Raspberry Pi eduroam guides online are written for Bullseye and earlier and do not apply.

```bash
systemctl is-active NetworkManager            # active
nmcli -t -f NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY connection show
nmcli -t -f NAME,DEVICE,TYPE connection show --active
nmcli device wifi list
```

Connection profiles are plaintext keyfiles in `/etc/NetworkManager/system-connections/`, mode `0600` and root-owned. NetworkManager **ignores any file in that directory that is readable or writable by a non-root user**, which is the usual reason a hand-written profile appears to do nothing.

To list stored networks without printing secrets:

```bash
sudo grep -H -E '^(id|ssid|key-mgmt)=' \
  /etc/NetworkManager/system-connections/*.nmconnection
```

### 2.2 Campus WiFi: eduroam via easyroam

The university uses **easyroam** (DFN). easyroam authenticates with **EAP-TLS and a per-device client certificate** — not PEAP with a username and password. There is no password to enter and no RADIUS server name to look up; the downloaded bundle contains the client certificate, the private key and the root CA.

| Property | Value |
| --- | --- |
| Realm | `uni-oldenburg.de` |
| Method | EAP-TLS, WPA2 802.1X |
| Identity | taken verbatim from the certificate Common Name; must not be edited |
| Scope | one certificate per device — do not reuse the laptop's on the Pi |

Two prerequisites, both of which produce misleading errors when wrong.

Regulatory domain, or the Pi cannot see 5 GHz APs at all:

```bash
sudo raspi-config nonint do_wifi_country DE
iw reg get | head -5                          # expect: country DE
```

System clock. A Pi 5 with no RTC battery boots with a stale clock after a long shutdown, and certificate validation then fails with an error that reads like a certificate fault:

```bash
timedatectl                                   # System clock synchronized: yes
```

### 2.3 Generating and installing the easyroam certificate

**Step 1 — generate.** On the laptop, log in at [easyroam.de](https://www.easyroam.de) through the DFN-AAI picker, then **Manual installation → Linux**. Name the profile identifiably (the Pi uses `HIL-RPi`) so it can be revoked individually later from the Profiles page. Download the PKCS12 (`.p12`) file.

**Step 2 — copy it over.** Run this in a Windows terminal on the laptop, *not* inside the SSH session:

```
scp "C:\Users\Khuzaim\Downloads\HIL-RPi.p12" ppre_hil@100.126.2.121:~/
```

**Step 3 — extract.** The bundle has an empty passphrase, and OpenSSL 3 on Bookworm needs `-legacy` to read it. Run all of this in one shell session, because `PKPW` is needed at the end:

```bash
mkdir -p ~/.cert/easyroam && cd ~/.cert/easyroam
mv ~/HIL-RPi.p12 . && P12=HIL-RPi.p12

openssl pkcs12 -in "$P12" -legacy -nokeys -passin pass: \
  | openssl x509 > easyroam_client_cert.pem

PKPW=$(tr -dc 'a-zA-Z0-9' </dev/urandom | head -c 20)
openssl pkcs12 -legacy -in "$P12" -nodes -nocerts -passin pass: \
  | openssl rsa -aes256 -passout pass:"$PKPW" -out easyroam_client_key.pem

openssl pkcs12 -in "$P12" -legacy -cacerts -nokeys -passin pass: \
  > easyroam_root_ca.pem

chmod 600 easyroam_client_key.pem "$P12"
CN=$(openssl x509 -noout -subject -in easyroam_client_cert.pem \
  | sed -n 's/^.*CN *= *\([^,]*\).*$/\1/p')
echo "identity: $CN"
openssl x509 -noout -dates -in easyroam_client_cert.pem
```

> **Gotcha.** `-legacy` belongs to `openssl pkcs12`, not to `openssl rsa`. Passing it to `rsa` alongside `-aes256` fails with `Multiple cipher or unknown options` and the key is never written.

**Step 4 — verify** before building the profile:

```bash
openssl rsa -noout -check -in easyroam_client_key.pem -passin pass:"$PKPW"
openssl x509 -noout -subject -issuer -in easyroam_root_ca.pem
```

The key should report `RSA key ok`. If the CA line errors, the `-cacerts` output carries bag-attribute text before the PEM block; strip it with:

```bash
sed -n '/BEGIN CERTIFICATE/,/END CERTIFICATE/p' easyroam_root_ca.pem \
  > ca.tmp && mv ca.tmp easyroam_root_ca.pem
```

**Step 5 — create the profile:**

```bash
sudo nmcli connection add type wifi ifname wlan0 con-name easyroam ssid eduroam \
  wifi-sec.key-mgmt wpa-eap \
  802-1x.eap tls \
  802-1x.identity '775287357419001210@easyroam-pca.uni-oldenburg.de' \
  802-1x.client-cert  /home/ppre_hil/.cert/easyroam/easyroam_client_cert.pem \
  802-1x.ca-cert      /home/ppre_hil/.cert/easyroam/easyroam_root_ca.pem \
  802-1x.private-key  /home/ppre_hil/.cert/easyroam/easyroam_client_key.pem \
  802-1x.private-key-password "$PKPW" \
  wifi.cloned-mac-address permanent \
  wifi.powersave 2 \
  connection.autoconnect yes \
  connection.autoconnect-priority 20
```

Absolute paths, not `$HOME` — `sudo` resets `HOME` to root's. `PKPW` does not need storing anywhere else; it lives in the root-only connection file.

Confirmed working, 12 October 2026: eduroam on 5540 MHz, WPA2 802.1X, DHCP address `10.19.1.156/20`.

### 2.4 Certificate renewal

> **Expires 12 October 2027.** When the certificate expires the Pi drops off eduroam silently and the rescue hotspot in [2.6](#26-rescue-hotspot--read-this-before-a-lab-visit) becomes the only way in. Set a calendar reminder for late September 2027.

Renewal is the same procedure as 2.3: generate a fresh PKCS12 from the portal, repeat the extractions, then point the existing profile at the new files rather than creating a second one:

```bash
sudo nmcli connection modify easyroam \
  802-1x.identity '<new CN>' \
  802-1x.private-key-password "$PKPW"
sudo nmcli connection up easyroam
```

Revoke the old certificate on the easyroam Profiles page afterwards.

### 2.5 Stored network profiles and priority

NetworkManager activates the highest-priority profile whose SSID is **currently visible**. It does not wait and fall back in sequence — an out-of-range network is skipped immediately. Higher number wins.

| Profile | SSID / purpose | Priority |
| --- | --- | --- |
| `easyroam` | `eduroam` — campus, EAP-TLS | 20 |
| `laptop-hotspot` | Windows Mobile Hotspot — rescue path, see 2.6 | 18 |
| `S-A52s` | `SA52s` — phone hotspot, second fallback | 15 |
| `dorm-wifi-1` | `SW-AP-3` — dorm, WPA2-Enterprise | 10 |
| `dorm-wifi-2` | `SW-AP-3a` — dorm, WPA2-Enterprise | 10 |

```bash
sudo nmcli connection modify <name> connection.autoconnect-priority <n>
```

### 2.6 Rescue hotspot — read this before a lab visit

If eduroam glitches, which it does regularly, the Pi falls back automatically to any visible hotspot matching the `laptop-hotspot` profile. **Nobody needs to touch the Pi.** Anyone can provide that hotspot from a laptop or a phone by setting it to exactly these values:

| Setting | Value |
| --- | --- |
| Network name | `DESKTOP-FAFMQMM 2864` |
| Password | `=1f79Y18` |
| Band | Any available |
| Security | WPA2-PSK |

Turn the hotspot on, wait about a minute for the Pi to scan and associate, then connect over Tailscale as in 2.7.

Two things to know about Windows Mobile Hotspot: it switches itself off after roughly five minutes with no client connected, so enable it *after* the Pi is already booted; and when the laptop is on 5 GHz the hotspot is forced to 5 GHz too, which is fine because the regulatory domain is set to DE.

### 2.7 SSH access

Use Tailscale. It is address-independent and works through eduroam, which isolates clients and blocks inbound connections, so a local-IP SSH from the laptop generally will not work on campus.

```bash
ssh ppre_hil@100.126.2.121
```

Checks from the Pi side:

```bash
tailscale status
nmcli -f ACTIVE,SSID,SIGNAL,FREQ,SECURITY device wifi | grep '^yes'
ip -4 addr show wlan0
```

> **Obsolete.** The address `192.168.137.138` recorded in the original log was a DHCP lease from the Windows Mobile Hotspot. It is not stable and should not be used. Hostname resolution via `hil-rpi.local` is also unreliable from Windows.

### 2.8 WiFi power management fix

The RPi WiFi adapter sleeps to save power and drops connections. This is critical — a dropped connection mid-timeseries run kills the Arduino serial loop. The file does not exist by default; nano creates a blank buffer.

```bash
sudo nano /etc/NetworkManager/conf.d/default-wifi-powersave-on.conf
```

Contents:

```ini
[connection]
wifi.powersave = 2
```

`wifi.powersave` values: **2** = disable power saving (always awake). **3** = enable (the problematic default). **0** = OS default. **1** = ignore kernel setting.

Each connection profile created above also sets `wifi.powersave 2` explicitly, so the setting survives even if the `conf.d` file is lost.

```bash
sudo reboot
```

### 2.9 Historical note — superseded

The original June 2026 log recorded the following, kept here for context but no longer describing the machine:

- *"Eduroam uses WPA2-Enterprise (802.1X) and must be configured manually via wpa_supplicant.conf."* Wrong on both counts: Bookworm uses NetworkManager, and the university uses easyroam with EAP-TLS certificates rather than manual 802.1X.
- Hostel WiFi — failed at the time, device-to-device communication blocked by the router. Two dorm profiles now exist and work.
- Phone hotspot — connected but `.local` mDNS resolution failed on Windows. Resolved by using Tailscale instead of mDNS.
- Laptop Windows Mobile Hotspot at `192.168.137.138`. The profile was later found to be **missing from the Pi entirely**, which cost an afternoon in October 2026. It now exists as `laptop-hotspot`.

> **Lesson.** A profile that is not in `/etc/NetworkManager/system-connections/` does not exist, no matter what the setup log says. Verify with `nmcli` before relying on a network as a fallback.

---

## 3. System Update

After first SSH login, update all system packages:

```bash
sudo apt update && sudo apt full-upgrade -y
```

---

## 4. Python Environment

### 4.1 Python Version Check

```bash
python3 --version        # Python 3.13.5
```

### 4.2 Virtual Environment

All project packages go inside an isolated virtual environment to avoid conflicts with system Python:

```bash
python3 -m venv hil_env
source hil_env/bin/activate
```

> **Important.** Run `source hil_env/bin/activate` every time you SSH in, before running any project code. The `(hil_env)` prefix in the prompt confirms activation. To leave it: `deactivate`.

### 4.3 Python Packages

```bash
source hil_env/bin/activate
pip install -r requirements.txt
```

The repository carries a complete `requirements.txt` including plotly, numba and cyipopt. Prefer it over installing packages individually. Verification:

```bash
python3 -c "import pandapower, simbench, serial; print('all good')"
```

---

## 5. Julia Installation

> **Important.** The Julia runtime is installed system-wide via juliaup, *not* inside `hil_env`. `juliacall` (the Python bridge) lives inside `hil_env`. This is correct and intentional — Julia is a runtime, not a Python package.

```bash
curl -fsSL https://install.julialang.org | sh
source ~/.bashrc
juliaup add 1.10.4
juliaup default 1.10.4
julia --version                 # julia version 1.10.4
```

`juliacall` verification, inside `hil_env`:

```bash
python3 -c "import juliacall; print('juliacall ok')"
```

---

## 6. cyipopt Installation (OPF Solver)

cyipopt wraps the IPOPT nonlinear optimisation solver. `pip install cyipopt` alone fails because it requires the IPOPT C library at the system level first.

> **Required for every run, not only the OPF.** `benchmark_runner.py` imports all six scenario runners at module level, and `scenario_5_opf.py` imports cyipopt unconditionally. Any CLI or direct-script run therefore needs it, even a scenario comparison that never touches the OPF.

### 6.1 Check available packages

The package name changed between Debian versions:

```bash
apt search coinor-libipopt
```

On Bookworm stable the runtime library is `coinor-libipopt3`, not `coinor-libipopt1v5` (an older Debian name).

### 6.2 Install system dependencies

```bash
sudo apt install build-essential pkg-config python3-dev cython3 \
  coinor-libipopt3 coinor-libipopt-dev
```

### 6.3 Install cyipopt via pip

```bash
source hil_env/bin/activate
pip install cyipopt
```

---

## 7. Disk Usage Check

```bash
df -h /
```

At initial setup: 11 GB used / 29 GB total / 17 GB free (40%). Monitor after the cyipopt install, as build tools add ~200 MB. Annual run outputs are large — checkpoint JSONL is written every timestep — so check free space before a long run.

---

## 8. Arduino Serial Setup

### 8.1 Device Detection

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

`/dev/ttyACM0` confirmed. `ttyUSB` not found, which is expected: `ttyUSB` is for USB-to-serial adapters and the Uno R3 uses native USB.

### 8.2 Serial Port Permissions

```bash
groups ppre_hil          # dialout present — no manual usermod required
```

### 8.3 Serial Communication Test

```bash
python3 -c "import serial; s = serial.Serial('/dev/ttyACM0', 115200, timeout=2); \
  print('Arduino connected:', s.isOpen()); s.close()"
```

Result: `Arduino connected: True`

---

## 9. Getting the Code onto the RPi

### 9.1 Git clone (preferred)

The repository is cloned directly on the Pi so updates arrive with `git pull` instead of scp. See [section 14](#14-keeping-the-rpi-in-sync-with-gitlab) for the one-time setup and the routine update command.

### 9.2 SCP (for files not in the repo)

SCP transfers files over the same SSH connection. Run all `scp` commands from a Windows terminal on the laptop — **not** from inside the SSH session. Run inside SSH, the RPi tries to copy from itself and fails with `ssh: Could not resolve hostname c: Name or service not known`.

The code itself comes from git, so scp is only for files that are not in the repository — an easyroam `.p12`, a weather CSV, a run output pulled back for analysis.

To the Pi:

```
scp "D:\My Files\data_OBS_DEU_PT10M_RADG_691.csv" ^
  ppre_hil@100.126.2.121:~/hil_project/data/dwd/
```

Back to the laptop:

```
scp ppre_hil@100.126.2.121:~/results.json C:\Users\Khuzaim\Desktop\
```

Use quotes around paths with spaces. `-r` copies directories recursively. Addresses use the Tailscale IP so the commands work from anywhere.

### 9.3 Exiting SSH

```bash
exit     # or Ctrl+D — RPi keeps running, only the terminal session closes
```

---

## 10. Running the Benchmark Script

```bash
source ~/hil_env/bin/activate
cd ~/hil_project
python run_benchmark_script.py
```

**Path note.** The script uses `parent.parent` path resolution. It must remain at the repository root (`~/hil_project`) — do not move it into a subfolder.

**dry_run=True.** Default setting in `BenchmarkConfig`. Runs all scenarios without Arduino hardware. Set to `False` and uncomment `port='/dev/ttyACM0'` for live HIL.

**Long runs.** Run inside `tmux`. A full annual run takes hours and a dropped SSH session would otherwise kill it. Crash-resume checkpointing covers a power loss, but not a scenario that never started.

---

## 11. Linux Command Reference

### 11.1 Navigation

| Command | Description |
| --- | --- |
| `pwd` | Print current directory |
| `ls` | List files and folders |
| `ls -la` | List all files including hidden, with permissions and sizes |
| `cd foldername` | Go into a folder |
| `cd ..` | Up one level |
| `cd ~` | Home directory (`/home/ppre_hil`) |
| `cd ~/hil_project` | Straight to the project folder |

### 11.2 File & Folder Management

| Command | Description |
| --- | --- |
| `mkdir foldername` | Create a folder |
| `rm filename.py` | Delete a file (no undo) |
| `rm -rf foldername` | Delete a folder and contents (no undo — careful) |
| `cp file.py copy.py` | Copy a file |
| `mv file.py newname.py` | Rename or move |
| `cat filename.py` | Print file contents |
| `nano filename.py` | Edit (Ctrl+X to exit, Y to save) |

### 11.3 Python & Virtual Environment

| Command | Description |
| --- | --- |
| `source ~/hil_env/bin/activate` | Activate the venv |
| `deactivate` | Leave the venv |
| `pip install -r requirements.txt` | Install the full dependency set |
| `pip list` | List installed packages |
| `python3 --version` | Python version |
| `which python3` | Confirms the venv is active |

### 11.4 System & Utilities

| Command | Description |
| --- | --- |
| `df -h /` | Disk usage |
| `free -h` | RAM usage |
| `top` | Live process monitor (q to quit) |
| `sudo apt update` | Refresh package lists |
| `sudo apt install pkg` | Install a system package |
| `sudo reboot` | Reboot (SSH session closes) |
| `timedatectl` | Clock and NTP sync — required before 802.1X |
| `history` | Command history |
| `Ctrl + C` | Stop a running command |
| `Ctrl + D` | Exit SSH session |
| `apt search name` | Search for an apt package |

### 11.5 Serial & Hardware

| Command | Description |
| --- | --- |
| `ls /dev/ttyACM*` | Is the Arduino detected? |
| `groups ppre_hil` | Confirm `dialout` membership |
| `dmesg \| tail -20` | Last 20 kernel messages — confirms detection after plugging in |

### 11.6 Networking

| Command | Description |
| --- | --- |
| `nmcli connection show` | List all stored profiles |
| `nmcli connection show --active` | Which profile is connected |
| `nmcli device wifi list` | Scan for visible networks |
| `nmcli connection up <name>` | Activate a profile |
| `nmcli networking connectivity check` | full / limited / none (needs root) |
| `iw reg get` | Regulatory domain — must be DE for 5 GHz |
| `tailscale status` | Tailscale peers and connection type |
| `journalctl -u wpa_supplicant -b` | 802.1X and EAP-TLS failures |
| `journalctl -u NetworkManager -b` | Association, DHCP, profile activation |

### 11.7 Git

| Command | Description |
| --- | --- |
| `git status` | Local changes vs the last commit |
| `git pull` | Fetch and merge the latest from GitLab |
| `git log --oneline -5` | Last five commits |
| `git stash` | Shelve local changes so `git pull` can proceed |
| `git clean -nd` | Preview which untracked files would be removed |

---

## 12. Confirmed Stack Summary

| Item | Value |
| --- | --- |
| OS | Raspberry Pi OS Lite (64-bit), Bookworm, kernel 6.12.75, aarch64 |
| Hostname / user | `HIL-RPi` / `ppre_hil` |
| Python | 3.13.5 inside `hil_env` |
| Packages | pandapower, simbench, pyserial, pvlib, juliacall, rich, matplotlib, plotly, numba, cyipopt |
| Julia | 1.10.4 via juliaup (system-wide) |
| Arduino Uno R3 | `/dev/ttyACM0`, 115200 baud, confirmed |
| Network stack | NetworkManager (not wpa_supplicant); regulatory domain DE |
| Campus WiFi | eduroam via easyroam, EAP-TLS, certificate expires **12 Oct 2027** |
| SSH | Tailscale at `100.126.2.121` — address-independent |
| Rescue path | hotspot `DESKTOP-FAFMQMM 2864`, see 2.6 |
| WiFi power management | Disabled (`powersave = 2`) |
| Storage | 11 GB used / 29 GB total at initial setup |
| Project files | `~/hil_project/` — git clone of the GitLab repo, updated with `git pull` |

---

## 13. Next Steps

- ~~Configure eduroam on the RPi~~ — **done, 12 October 2026**, see section 2
- Upload `volt_var_arduino.ino` to the Uno R3 via Arduino IDE and run the HIL handshake test
- Run `volt_var_controller.py` with `dry_run=True` to confirm offline operation
- Run the full HIL loop with SimBench `1-MV-rural--2-sw` (Scenario 4, `dry_run=False`, `port=/dev/ttyACM0`)
- Full annual re-run on dry-run laptop and HIL RPi, then re-run `analyze_timing.py`, `diagnose_outliers.py` and `compare_scenarios.py`
- Calendar reminder for late September 2027 to renew the easyroam certificate

---

## 14. Keeping the RPi in Sync with GitLab

One-time setup, then `git pull` replaces scp for all code updates.

### 14.1 Install git

```bash
git --version
```

If that reports `command not found`:

```bash
sudo apt update && sudo apt install -y git
```

### 14.2 SSH key for GitLab

HTTPS would mean storing a token in plaintext in `~/.git-credentials`. An SSH key is cleaner for a headless, shared machine.

```bash
ssh-keygen -t ed25519 -C "HIL-RPi" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

Copy that public key into GitLab → **Preferences → SSH Keys → Add new key**. Title it `HIL-RPi` so it can be revoked separately. Then test:

```bash
ssh -T git@gitlab.uni-oldenburg.de
```

A greeting by username means it worked. If the connection hangs or is refused, outbound port 22 is blocked on that network — use the hotspot, or fall back to HTTPS with a read-only deploy token.

### 14.3 Clone

The repository lives **directly at `~/hil_project`** — repo contents at the top level, not inside a subfolder. On a fresh machine, clone straight into that path:

```bash
git clone git@gitlab.uni-oldenburg.de:sebi0767/hil-tesbed.git ~/hil_project
cd ~/hil_project
git status
git log --oneline -3
```

`git clone <url> <dir>` requires `<dir>` to be absent or empty. If an older copy is in the way, move it aside first and delete it only once any run outputs worth keeping have been copied out:

```bash
mv ~/hil_project ~/hil_project.old
du -sh ~/hil_project.old/*          # check before deleting
rm -rf ~/hil_project.old
```

> **Moving an existing clone.** Move the *directory itself*, never its contents. `mv src/* dst/` silently leaves `.git` and `.gitignore` behind and turns a clone into an untracked folder. If contents must be moved, `shopt -s dotglob` first. Verify with `git status` afterwards — "not a git repository" means `.git` was left behind.

### 14.4 Routine update

```bash
cd ~/hil_project
git pull
source ~/hil_env/bin/activate
pip install -r requirements.txt     # only when dependencies changed
```

`runs/` and `outputs/` are gitignored, so local run results are never touched by a pull.

### 14.5 If git pull refuses

A pull fails when local files conflict with incoming changes. Check what is modified, then shelve or discard:

```bash
git status
git stash                 # shelve local edits, then: git pull && git stash pop
git checkout -- <file>    # discard edits to one file
```

Do not commit from the Pi unless you mean to. If you do, set an identity first:

```bash
git config --global user.name  "Khuzaim Khan"
git config --global user.email "khuzaimkhan234@gmail.com"
```
