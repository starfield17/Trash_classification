#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

int is_module_loaded(const char *module_name) {
    FILE *fp;
    char line[256];
    int found = 0;
    
    fp = popen("lsmod", "r");
    if (fp == NULL) {
        perror("Failed to run lsmod");
        return -1;
    }
    
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strstr(line, module_name)) {
            found = 1;
            break;
        }
    }
    
    pclose(fp);
    return found;
}

void load_module(const char *module_name) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "modprobe %s", module_name);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "Failed to load module %s\n", module_name);
    } else {
        printf("Module %s loaded successfully\n", module_name);
    }
}

void unload_module(const char *module_name) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "rmmod %s 2>/dev/null", module_name);
    system(cmd);
}

void write_to_file(const char *filename, const char *value) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Failed to open file for writing");
        return;
    }
    fprintf(fp, "%s", value);
    fclose(fp);
}
void rescan_usb_devices() {
    DIR *dir;
    struct dirent *entry;
    char path[512];
    char authorized_path[512];
    char real_path[512];
    
    dir = opendir("/sys/bus/usb/devices");
    if (!dir) {
        perror("Failed to open directory");
        return;
    }
    
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "usb", 3) == 0) {
            snprintf(path, sizeof(path), "/sys/bus/usb/devices/%s", entry->d_name);
            
            // 获取实际路径
            if (realpath(path, real_path) == NULL) {
                perror("Failed to get real path");
                continue;
            }
            snprintf(authorized_path, sizeof(authorized_path), "%s/authorized", real_path);
            struct stat st;
            if (stat(authorized_path, &st) == 0) {
                printf("Rescanning: %s\n", real_path);
                write_to_file(authorized_path, "0");
                sleep(1);
                write_to_file(authorized_path, "1");
            }
        }
    }
    
    closedir(dir);
}

int main() {
    if (geteuid() != 0) {
        fprintf(stderr, "This program must be run as root.\n");
        return 1;
    }
    printf("Starting USB rescanning program...\n");
    sleep(5);
    if (!is_module_loaded("ch341")) {
        printf("Loading ch341 module...\n");
        load_module("ch341");
    }
    
    rescan_usb_devices();
    
    struct stat st;
    if (stat("/dev/ttyUSB0", &st) != 0) {
        printf("Device /dev/ttyUSB0 not detected, reloading usb driver...\n");
        unload_module("ch341");
        sleep(1);
        load_module("ch341");
    }
    
    printf("USB device scan completed\n");
    return 0;
}

