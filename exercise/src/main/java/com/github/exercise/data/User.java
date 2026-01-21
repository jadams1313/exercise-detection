package com.github.exercise.data;

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
    private String passwordHash;

    @Column(name="region", nullable=true)
    private String region;


    public Long getId() {
        return id;
    }

    public String getUserName() {
        return userName;
    }

    public LocalDateTime getCreateDateTime() {
        return createDateTime;
    }

    public String getFirstName() {
        return firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public String getEmail() {
        return email;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public LocalDateTime getUpdatedDateTime() {
        return updatedDateTime;
    }

    public String getPasswordHash() {
        return passwordHash;
    }

    public String getRegion() {
        return region;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public void setUserName(String userName) {
        this.userName = userName;
    }

    public void setCreateDateTime(LocalDateTime createDateTime) {
        this.createDateTime = createDateTime;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    public void setUpdatedDateTime(LocalDateTime updatedDateTime) {
        this.updatedDateTime = updatedDateTime;
    }

    public void setPasswordHash(final String passwordHash) {
        this.passwordHash = passwordHash;
    }

    public void setRegion(String region) {
        this.region = region;
    }
}
