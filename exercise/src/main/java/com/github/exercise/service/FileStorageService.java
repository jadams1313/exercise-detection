package com.github.exercise.service;

import com.github.exercise.util.S3Util;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.util.UUID;

import static reactor.netty.http.HttpConnectionLiveness.log;

@Slf4j
@Service
@RequiredArgsConstructor
public class FileStorageService {

    private final S3Util s3Util;

    public String storeFile(MultipartFile file) throws IOException {
        String originalFilename = file.getOriginalFilename();
        String fileExtension = originalFilename != null && originalFilename.contains(".")
                ? originalFilename.substring(originalFilename.lastIndexOf("."))
                : "";

        String storedFileName = UUID.randomUUID().toString() + fileExtension;

        s3Util.putObject(storedFileName, file.getInputStream(), file.getSize(), file.getContentType());
        log.debug("File stored to S3: {}", storedFileName);

        return storedFileName;
    }

    public InputStream getFileStream(String fileName) {
        return s3Util.getObject(fileName);
    }

    public void deleteFile(String fileName) {
        s3Util.deleteObject(fileName);
        log.debug("File deleted from S3: {}", fileName);
    }

    public String getPresignedUrl(String fileName, int expirationMinutes) {
        return s3Util.generatePresignedUrl(fileName, expirationMinutes);
    }
}