package com.github.data;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;
import org.springframework.data.annotation.CreatedDate;

import java.time.LocalDateTime;

@Entity
@Table(name = "user")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name="user_name", updatable=true, nullable=false)
    private String userName;

    @CreationTimestamp
    @Column(name="created_date_time", updatable = false, nullable=false)
    private LocalDateTime createDateTime;

    @Column(name = "first_name", nullable = true)
    private String firstName;

    @Column(name = "last_name", nullable=true)
    private String lastName;

    @Column(name = "email", nullable=true)
    private String email;

    @Column(name = "phone_number", nullable=true)
    private String phoneNumber;

    @Column(name = "updated_date_time", nullable=true)
    @UpdateTimestamp
    private LocalDateTime  updatedDateTime;

    @Column(name="password", nullable=true)
    private String password;

}
