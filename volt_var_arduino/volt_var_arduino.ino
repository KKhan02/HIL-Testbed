/*
 * volt_var_arduino.ino
 * =====================
 * Phase 1 Item 2 -- Q(V) Volt-Var controller (Arduino side).
 *
 * Implements the VDE-AR-N 4110 Bild 8 Q(U) piecewise linear characteristic
 * and the serial exchange protocol for the HIL testbed.
 *
 * Hardware
 * --------
 * Target  : Arduino Uno R3 (ATmega328P, 32 KB flash, 2 KB SRAM)
 * Connect : USB cable to Raspberry Pi 5 (/dev/ttyACM0 or /dev/ttyUSB0)
 *
 * Q(U) Characteristic -- VDE-AR-N 4110 Bild 8
 * ---------------------------------------------
 * Constants below MUST match volt_var_controller.py exactly.
 *
 *   U <= 0.96 pu   =>  Q = +Q_max          (inject, raise voltage)
 *   0.96-0.99 pu   =>  linear +Q_max -> 0
 *   0.99-1.01 pu   =>  Q = 0               (deadband)
 *   1.01-1.04 pu   =>  linear 0 -> -Q_max
 *   U >= 1.04 pu   =>  Q = -Q_max          (absorb, lower voltage)
 *
 *   Q_max = Q_RATIO x P_b_inst (MW)
 *
 * Sign: positive Q = inject (raises V), negative Q = absorb (lowers V).
 * Matches pandapower sgen.q_mvar convention.
 *
 * Serial Protocol  (ASCII, 115200 baud, newline-terminated)
 * --------------------------------------------------------
 * Startup handshake:
 *   RPi  -> Arduino :  "INIT:<n>\n"          n = number of DERs
 *   Arduino -> RPi  :  "ACK:INIT\n"
 *   RPi  -> Arduino :  "P:<p1>,<p2>,...\n"   p_installed per DER (MW)
 *   Arduino -> RPi  :  "ACK:P\n"
 *
 * Per timestep:
 *   RPi  -> Arduino :  "V:<v1>,<v2>,...\n"   vm_pu per DER
 *   Arduino -> RPi  :  "Q:<q1>,<q2>,...\n"   q_mvar per DER
 *
 * Error responses:
 *   "ERR:INIT_RANGE"      n <= 0 or n > MAX_DERS
 *   "ERR:P_BEFORE_INIT"   P message received before valid INIT
 *   "ERR:P_COUNT"         wrong number of P values
 *   "ERR:NOT_CONFIGURED"  V message before P handshake
 *   "ERR:V_COUNT"         wrong number of V values
 *   "ERR:V_INVALID"       NaN or Inf in received vm_pu
 *   "ERR:RESP_OVERFLOW"   Q: response would overflow BUF_SIZE
 *   "ERR:BUF_OVERFLOW"    incoming message exceeds BUF_SIZE without newline
 *   "ERR:UNKNOWN"         message prefix not recognised
 *
 * Memory Budget  (ATmega328P, 2 KB SRAM)
 * ----------------------------------------
 * Global state:
 *   p_installed[MAX_DERS]    : 105 x 4 = 420 bytes
 *   vm_buf[MAX_DERS]         : 105 x 4 = 420 bytes
 *   buf[BUF_SIZE]            : 960 bytes
 *   n_ders, configured,
 *   skip_to_newline, buf_idx : ~6 bytes
 *   Total globals            : ~1806 bytes
 *   Remaining for stack      : ~242 bytes  (2048 - 1806)
 *   Estimated stack usage    : ~80 bytes   (handle_message + loop frames)
 *   Free margin              : ~162 bytes  -- do not increase MAX_DERS
 *                              without recalculating.
 *
 * Response buffer sizing (BUF_SIZE = 960):
 *   Worst case per DER: "-XXXX.DDDD" = 10 chars (trimmed by dtostrf strip)
 *   Typical MV DER:     "X.DDDD" or "-X.DDDD" = 6--7 chars
 *   Typical 105 DERs:   2 + 105*(7+1) - 1 + 1 = 842 bytes  (< 960) OK
 *   Pathological case:  2 + 105*(10+1) - 1 + 1 = 1157 bytes (> 960)
 *                       triggers ERR:RESP_OVERFLOW; not realistic for
 *                       any MV/LV network with normal installed capacities.
 *   BUF_SIZE=960 provides ~118 bytes headroom for typical outputs.
 *   To increase MAX_DERS: BUF_SIZE >= 2 + MAX_DERS*(8+1) + 2 and verify
 *   total globals remain under ~1900 bytes (leaves ~148 bytes for stack).
 *
 * tmp buffer sizing (tmp[TMP_SIZE] = tmp[20]):
 *   dtostrf(val, width, prec, buf) writes at most width+1 chars (with null).
 *   FLOAT_WIDTH=8 means minimum 8-char field, but very large Q values (e.g.
 *   "-9999.0000" = 10 chars) exceed this. tmp[20] is safe for any realistic
 *   MVAr value and provides a margin for edge cases.
 *
 * Why char arrays instead of String objects?
 *   The Arduino String class allocates on the heap. Repeated String creation
 *   and destruction during a long HIL run (hours) causes heap fragmentation
 *   on the 2 KB ATmega328P SRAM, eventually producing malloc failures and
 *   undefined behaviour. char arrays have zero heap involvement -- they live
 *   entirely on the stack or in the global static section.
 *
 * Why dtostrf instead of snprintf("%f")?
 *   The AVR-libc snprintf %f conversion requires linking the floating-point
 *   printf variant, which adds ~1.5 KB to flash. More critically, some AVR
 *   toolchains silently skip %f and print nothing. dtostrf is the idiomatic
 *   AVR approach for float-to-string conversion.
 *
 * Notes
 * -----
 * - Arduino Uno R3 resets on USB serial open. RPi waits ARDUINO_RESET_DELAY_S
 *   (2 s) after open(), then calls reset_input_buffer() before configure().
 * - "READY\n" signals boot completion on the serial monitor (development aid).
 *   RPi discards it via reset_input_buffer() and configure()'s _read_ack().
 */

#include <math.h>      /* isnan(), isinf(), fabsf() */
#include <stdlib.h>    /* strtof(), strtol(), atoi() -- explicit for non-Arduino-IDE builds */
#include <string.h>    /* strncmp(), memcpy(), strlen() */

// ===========================================================================
// Q(U) characteristic constants -- mirror volt_var_controller.py exactly
// ===========================================================================

#define U1_PU    0.96f
#define U2_PU    0.99f
#define U3_PU    1.01f
#define U4_PU    1.04f
#define Q_RATIO  0.48f

// ===========================================================================
// Protocol / buffer constants
// ===========================================================================

#define BAUD_RATE    115200
#define MAX_DERS     105     /* maximum controllable DERs */
#define BUF_SIZE     960    /* serial receive/response buffer; see Memory Budget */
#define FLOAT_WIDTH  8      /* dtostrf minimum field width */
#define FLOAT_DEC    4      /* dtostrf decimal places */
#define TMP_SIZE     20     /* dtostrf scratch buffer; safe for any realistic MVAr */
/* FLOAT_CHARS removed: max chars per float depends on value magnitude, not FLOAT_WIDTH.
   TMP_SIZE=20 covers the actual worst case. Use TMP_SIZE for sizing decisions. */

// ===========================================================================
// Global state
// ===========================================================================

static float    p_installed[MAX_DERS];  /* rated capacity per DER (MW) */
static float    vm_buf[MAX_DERS];     /* voltages parsed from V: — global to avoid 420-byte stack frame */
static int      n_ders         = 0;     /* number of DERs (set by INIT) */
static bool     configured     = false; /* true after successful P handshake */
static bool     skip_to_newline = false; /* set on BUF_OVERFLOW; discard tail bytes */

static char     buf[BUF_SIZE];        /* receive buffer AND response buffer (reused after V: parse) */
static uint16_t  buf_idx        = 0;     /* write position (must handle BUF_SIZE up to 960) */


// ===========================================================================
// Q(U) characteristic -- mirrors QVCharacteristic.compute_setpoint() exactly
// ===========================================================================

/*
 * compute_q()
 * -----------
 * Computes q_mvar for one DER given its connection bus voltage and rated
 * installed power capacity.
 *
 * Boundary conditions: inclusive (<= and >=) at saturation points U1 and U4.
 * Using strict < at U1_PU forces the ramp branch to execute a floating-point
 * division even when vm_pu is exactly U1_PU -- the result is identical to
 * q_max (the ramp evaluates to q_max*(U2-U1)/(U2-U1) = q_max) but the
 * division is unnecessary. Inclusive <= avoids this on the ATmega328P, which
 * has no hardware FPU and performs division in ~18-cycle software routines.
 *
 * Returns: q_mvar. Positive = inject, negative = absorb.
 */
float compute_q(float vm_pu, float p_inst_mw) {
    float q_max = Q_RATIO * fabsf(p_inst_mw);

    if (vm_pu <= U1_PU) {
        return q_max;
    } else if (vm_pu < U2_PU) {
        return q_max * (U2_PU - vm_pu) / (U2_PU - U1_PU);
    } else if (vm_pu <= U3_PU) {
        return 0.0f;
    } else if (vm_pu < U4_PU) {
        return -q_max * (vm_pu - U3_PU) / (U4_PU - U3_PU);
    } else {
        return -q_max;
    }
}


// ===========================================================================
// Float array parser
// ===========================================================================

/*
 * parse_float_array()
 * -------------------
 * Parses up to max_n comma-separated floats from a null-terminated string.
 * Uses strtof() for each token. strtof() advances the pointer past the
 * parsed token; end == ptr after a call means no characters were consumed
 * (malformed token) and parsing stops.
 *
 * Returns the number of floats successfully parsed. The caller checks that
 * this equals the expected count.
 *
 * Alternative: sscanf with repeated %f,%f -- but sscanf on AVR is expensive
 * in flash (~800 bytes) and does not give per-token error positions.
 */
int parse_float_array(char *data, float *out, int max_n) {
    int   n   = 0;
    char *ptr = data;
    char *end;

    while (n < max_n && *ptr != '\0') {
        out[n] = strtof(ptr, &end);
        if (end == ptr) break;
        n++;
        ptr = end;
        if (*ptr == ',') ptr++;
    }
    return n;
}


// ===========================================================================
// Message handler
// ===========================================================================

/*
 * handle_message()
 * ----------------
 * Dispatches a complete null-terminated message from the receive buffer.
 * Called from loop() when '\n' is detected.
 *
 * All responses use Serial.println() which appends "\r\n". The RPi's
 * readline() terminates on '\n'; the '\r' is stripped by .strip().
 */
void handle_message(char *msg) {

    /* ---- INIT:<n> ---- */
    if (strncmp(msg, "INIT:", 5) == 0) {
        /*
         * strtol with endptr rejects "INIT:5xyz" -- atoi would silently
         * return 5 and accept the malformed message. After parsing, endptr
         * must point to '\0'; any trailing non-digit means malformed input.
         */
        char *endptr;
        long n_long = strtol(msg + 5, &endptr, 10);
        int  n      = (int)n_long;

        if (endptr == msg + 5 || *endptr != '\0' || n <= 0 || n > MAX_DERS) {
            n_ders     = 0;
            configured = false;
            Serial.println("ERR:INIT_RANGE");
            return;
        }
        n_ders     = n;
        configured = false;
        Serial.println("ACK:INIT");
        return;
    }

    /* ---- P:<p1>,<p2>,... ---- */
    if (strncmp(msg, "P:", 2) == 0) {
        if (n_ders <= 0) {
            Serial.println("ERR:P_BEFORE_INIT");
            return;
        }
        int n = parse_float_array(msg + 2, p_installed, n_ders);
        if (n != n_ders) {
            Serial.println("ERR:P_COUNT");
            return;
        }
        configured = true;
        Serial.println("ACK:P");
        return;
    }

    /* ---- V:<v1>,<v2>,... ---- */
    if (strncmp(msg, "V:", 2) == 0) {
        if (!configured) {
            Serial.println("ERR:NOT_CONFIGURED");
            return;
        }

        int n = parse_float_array(msg + 2, vm_buf, n_ders);
        if (n != n_ders) {
            Serial.println("ERR:V_COUNT");
            return;
        }

        /*
         * Finiteness check on all vm_pu values before computing Q.
         * isnan() and isinf() are provided by <math.h> on AVR.
         * A NaN or Inf voltage (caused by serial corruption or a Python-
         * side bug) would propagate through compute_q() and produce a NaN
         * or Inf Q value in the response, which would fail float() on the
         * RPi and trigger a retry -- but explicit rejection here is more
         * informative and avoids wasting a compute_q() call.
         */
        for (int i = 0; i < n_ders; i++) {
            if (isnan(vm_buf[i]) || isinf(vm_buf[i])) {
                Serial.println("ERR:V_INVALID");
                return;
            }
        }

        /*
         * Build "Q:<q1>,<q2>,...\0" response.
         *
         * dtostrf(value, width, decimals, buffer):
         *   width   = minimum field width; dtostrf pads with leading spaces.
         *   buffer  = destination; must be at least width+1 bytes.
         *   TMP_SIZE = 20 is safe for any value from -9999999.9999 to
         *   +9999999.9999. Typical MV Q values are <<1 MVAr; this is
         *   defensive sizing only.
         *
         * Leading spaces are stripped before appending to resp[]. This
         * produces compact output ("0.4800" not "  0.4800") and keeps
         * the response within BUF_SIZE for all realistic DER counts.
         *
         * The overflow guard (pos + len + 2 >= BUF_SIZE) catches the
         * edge case where MAX_DERS is increased without also increasing
         * BUF_SIZE. The caller (RPi) will receive ERR:RESP_OVERFLOW,
         * which is safe and diagnosable.
         */
        /* Reuse buf[] as response buffer — V: message already parsed into vm_buf[],
         *buf[] is no longer needed as receive buffer. Avoids 960-byte stack frame. */
        char tmp[TMP_SIZE];
        int  pos = 0;

        buf[pos++] = 'Q';
        buf[pos++] = ':';
        buf[pos]   = '\0';

        for (int i = 0; i < n_ders; i++) {
            float q = compute_q(vm_buf[i], p_installed[i]);
            dtostrf(q, FLOAT_WIDTH, FLOAT_DEC, tmp);

            /* Strip leading spaces produced by dtostrf padding */
            char *t = tmp;
            while (*t == ' ') t++;

            int len = strlen(t);
            if (pos + len + 2 >= BUF_SIZE) {
                Serial.println("ERR:BUF_OVERFLOW");
                return;
            }
            memcpy(buf + pos, t, len);
            pos += len;

            if (i < n_ders - 1) {
                buf[pos++] = ',';
            }
        }
        buf[pos] = '\0';
        Serial.println(buf);
        return;
    }

    /* ---- Unknown message ---- */
    Serial.println("ERR:UNKNOWN");
}


// ===========================================================================
// Arduino lifecycle
// ===========================================================================

void setup() {
    Serial.begin(BAUD_RATE);
    /*
     * "READY\n" signals that setup() has completed and the loop is about
     * to start. The RPi discards it via reset_input_buffer() in open() and
     * the _read_ack() skip-loop in configure(). It is retained as a
     * development aid -- visible on the Arduino Serial Monitor.
     */
    Serial.println("READY");
}

void loop() {
    /*
     * Non-blocking character accumulation. Reads one byte per iteration of
     * the while loop, building a message in buf[]. On '\n', dispatches the
     * complete message and resets buf_idx.
     *
     * Why non-blocking?
     *   A blocking read (Serial.readStringUntil, Serial.readBytesUntil) stalls
     *   loop() for up to timeout_ms with no data incoming, freezing the MCU.
     *   Non-blocking returns immediately when the hardware UART buffer is empty.
     *
     * '\r' discarding:
     *   Windows terminal programmes send \r\n. Discarding '\r' avoids dispatching
     *   a second (empty) message when '\n' follows.
     *
     * Overflow and tail-corruption prevention (skip_to_newline flag):
     *   If buf fills without '\n' (runaway/malformed message), buf_idx is reset,
     *   ERR:BUF_OVERFLOW is sent, and skip_to_newline is set to true. While
     *   skip_to_newline is true, all incoming bytes are discarded until '\n'
     *   arrives. Without this, the Arduino would buffer the tail of the runaway
     *   message (bytes that arrived after the overflow), dispatch that tail to
     *   handle_message(), and produce spurious ERR:UNKNOWN responses.
     */
    while (Serial.available()) {
        char c = (char)Serial.read();

        if (c == '\n') {
            if (skip_to_newline) {
                /* Tail of an overflowed message -- discard silently */
                skip_to_newline = false;
                buf_idx = 0;
            } else {
                buf[buf_idx] = '\0';
                if (buf_idx > 0) {
                    handle_message(buf);
                }
                buf_idx = 0;
            }
        } else if (c == '\r') {
            /* discard Windows line-ending artifact */
        } else if (skip_to_newline) {
            /* discarding tail bytes of an overflowed message */
        } else if (buf_idx < BUF_SIZE - 1) {
            buf[buf_idx++] = c;
        } else {
            /* Buffer full without newline -- set flag, send error */
            skip_to_newline = true;
            buf_idx = 0;
            Serial.println("ERR:BUF_OVERFLOW");
        }
    }
}
