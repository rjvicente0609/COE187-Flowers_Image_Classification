/*******************************************************************************
* Copyright (C) 2019-2024 Maxim Integrated Products, Inc., All rights Reserved.
*
* Enhanced flowers with Camera, Button, and ASCII Art support
* Board: FTHR_REVA (Feather Board)
*******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "mxc_device.h"
#include "mxc_sys.h"
#include "icc.h"
#include "led.h"
#include "dma.h"
#include "pb.h"
#include "cnn.h"
#include "mxc_delay.h"
#include "camera.h"

// Comment out to disable ASCII art display
#define ASCII_ART

// Image dimensions (128x128 for flowers model)
#define IMAGE_SIZE_X (128)
#define IMAGE_SIZE_Y (128)

#define CAMERA_FREQ (5 * 1000 * 1000)

// Flower class names (update based on your trained model's classes)
const char classes[CNN_NUM_OUTPUTS][12] = {
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
};

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

volatile uint32_t cnn_time; // Stopwatch

// Buffer for camera image (128x128 = 16384 pixels)
static uint32_t input_0[IMAGE_SIZE_X * IMAGE_SIZE_Y];

/* **************************************************************************** */
#ifdef ASCII_ART
// ASCII brightness characters from dark to light
char *brightness = "@%#*+=-:. "; // simple
#define RATIO 2 // ratio of scaling down the image (2 = half size for 128x128)

void asciiart(uint8_t *img)
{
    int skip_x, skip_y;
    uint8_t r, g, b, Y;
    uint8_t *srcPtr = img;
    int l = strlen(brightness) - 1;

    skip_x = RATIO;
    skip_y = RATIO;
    
    printf("\n=== ASCII Art Representation ===\n");
    
    for (int i = 0; i < IMAGE_SIZE_Y; i++) {
        for (int j = 0; j < IMAGE_SIZE_X; j++) {
            // 0x00bbggrr, convert to [0,255] range
            r = *srcPtr++ ^ 0x80;
            g = *(srcPtr++) ^ 0x80;
            b = *(srcPtr++) ^ 0x80;
            srcPtr++; //skip msb=0x00

            // Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            Y = (3 * r + b + 4 * g) >> 3; // simple luminance conversion
            
            if ((skip_x == RATIO) && (skip_y == RATIO))
                printf("%c", brightness[l - (Y * l / 255)]);

            skip_x++;
            if (skip_x > RATIO)
                skip_x = 1;
        }
        skip_y++;
        if (skip_y > RATIO) {
            printf("\n");
            skip_y = 1;
        }
    }
    printf("================================\n\n");
}
#endif

/* **************************************************************************** */
void fail(void)
{
    printf("\n*** FAIL ***\n\n");
    while (1) {}
}

/* **************************************************************************** */
void cnn_load_input(void)
{
    int i;
    const uint32_t *in0 = input_0;

    for (i = 0; i < 16384; i++) { // 128x128 = 16384
        // Wait for FIFO 0
        while (((*((volatile uint32_t *)0x50000004) & 1)) != 0) {}
        *((volatile uint32_t *)0x50000008) = *in0++; // Write FIFO 0
    }
}

/* **************************************************************************** */
void capture_process_camera(void)
{
    uint8_t *raw;
    uint32_t imgLen;
    uint32_t w, h;
    int cnt = 0;
    uint8_t r, g, b;
    uint8_t *data = NULL;
    stream_stat_t *stat;

    printf("Starting camera capture...\n");
    camera_start_capture_image();

    // Get the details of the image from the camera driver.
    camera_get_image(&raw, &imgLen, &w, &h);
    printf("Camera: W=%d H=%d Length=%d\n", w, h, imgLen);

    // Get image line by line
    for (int row = 0; row < h; row++) {
        // Wait until camera streaming buffer is full
        while ((data = get_camera_stream_buffer()) == NULL) {
            if (camera_is_image_rcv()) {
                break;
            }
        }

        for (int k = 0; k < 4 * w; k += 4) {
            // data format: 0x00bbggrr
            r = data[k];
            g = data[k + 1];
            b = data[k + 2];
            //skip k+3

            // change the range from [0,255] to [-128,127] and store in buffer for CNN
            input_0[cnt++] = ((b << 16) | (g << 8) | r) ^ 0x00808080;
        }

        // Release stream buffer
        release_camera_stream_buffer();
    }

    stat = get_camera_stream_statistic();

    if (stat->overflow_count > 0) {
        printf("ERROR: Camera overflow detected = %d\n", stat->overflow_count);
        LED_On(LED2); // Turn on red LED if overflow detected
        while (1) {}
    }

    printf("Camera capture complete!\n");
}

/* **************************************************************************** */
int main(void)
{
    int i;
    int digs, tens;
    int ret = 0;
    int result[CNN_NUM_OUTPUTS];
    int dma_channel;
    int max_class = 0;

    // Wait for PMIC 1.8V to become available
    MXC_Delay(200000);

    printf("\n\n=================================\n");
    printf("Flowers Classification Demo\n");
    printf("=================================\n\n");

    /* Enable cache */
    MXC_ICC_Enable(MXC_ICC0);

    /* Switch to 100 MHz clock */
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    /* Enable peripheral, enable CNN interrupt, turn on CNN clock */
    /* CNN clock: 50 MHz div 1 */
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

    /* Configure P2.5, turn on the CNN Boost */
    cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5);

    /* Bring CNN state machine into consistent state */
    cnn_init();
    /* Load CNN kernels */
    cnn_load_weights();
    /* Load CNN bias */
    cnn_load_bias();
    /* Configure CNN state machine */
    cnn_configure();

    // Initialize DMA for camera interface
    printf("Initializing DMA...\n");
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    // Initialize camera
    printf("Initializing Camera...\n");
    camera_init(CAMERA_FREQ);

    ret = camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y, PIXFORMAT_RGB888, FIFO_THREE_BYTE,
                       STREAMING_DMA, dma_channel);
    if (ret != STATUS_OK) {
        printf("Error: Camera setup failed with error %d\n", ret);
        return -1;
    }

    // Set camera clock prescaler to prevent streaming overflow
    camera_write_reg(0x11, 0x0);
    printf("Camera initialized successfully!\n\n");

    printf("********** Press PB1(SW1) to capture an image **********\n");
    while (!PB_Get(0)) {}

    // Enable CNN clock
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);

    printf("\n*** CNN Inference Started ***\n\n");

    while (1) {
        LED_Off(LED1);
        LED_Off(LED2);

        // Capture image from camera
        capture_process_camera();

        printf("Starting CNN inference...\n");
        cnn_start();
        cnn_load_input();

        SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
        while (cnn_time == 0) {
            __WFI(); // Wait for CNN interrupt
        }

        // Unload CNN data
        cnn_unload((uint32_t *)ml_data);
        cnn_stop();

        // Softmax
        softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);

        printf("\nInference Time: %d us\n", cnn_time);
        printf("\n*** Classification Results ***\n");

        // Find the class with highest confidence
        max_class = 0;
        for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
            digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
            tens = digs % 10;
            digs = digs / 10;
            result[i] = digs;
            printf("[%7d] -> Class %d (%10s): %d.%d%%\n",
                   ml_data[i], i, classes[i], result[i], tens);

            if (ml_data[i] > ml_data[max_class]) {
                max_class = i;
            }
        }

        // Display winner and control LED
        printf("\n=== RESULT ===\n");
        printf("Detected: %s with %d.%d%% confidence\n",
               classes[max_class], result[max_class], tens);
        printf("==============\n");

        // LED indication: blink LED1 based on confidence
        if (result[max_class] > 80) {
            // High confidence: solid green
            LED_On(LED1);
            LED_Off(LED2);
        } else if (result[max_class] > 50) {
            // Medium confidence: both LEDs
            LED_On(LED1);
            LED_On(LED2);
        } else {
            // Low confidence: red LED
            LED_Off(LED1);
            LED_On(LED2);
        }

#ifdef ASCII_ART
        asciiart((uint8_t *)input_0);
#endif

        printf("\n********** Press PB1(SW1) to capture next image **********\n");
        while (!PB_Get(0)) {}
        printf("\n");
    }

    return 0;
}

/*
  SUMMARY OF OPS
  Hardware: 51,373,056 ops (50,436,096 macc; 936,960 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 7,340,032 ops (7,077,888 macc; 262,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 19,267,584 ops (18,874,368 macc; 393,216 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 19,070,976 ops (18,874,368 macc; 196,608 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 4,792,320 ops (4,718,592 macc; 73,728 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 600,064 ops (589,824 macc; 10,240 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 295,936 ops (294,912 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 6,144 ops (6,144 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 61,872 bytes out of 442,368 bytes total (14.0%)
  Bias memory:   6 bytes out of 2,048 bytes total (0.3%)
*/
