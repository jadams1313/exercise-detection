package com.github.data;

import com.github.constants.VideoStatus;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "video_upload")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class VideoUpload {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(nullable = true)
    private String filename;

    @Column(nullable = true)
    private String fileUrl;

    private Long fileSizeBytes;

    @Enumerated(EnumType.STRING)
    @Column(nullable = true)
    private VideoStatus status;

    @CreationTimestamp
    @Column(nullable = true, updatable = false)
    private LocalDateTime uploadedAt;

}
