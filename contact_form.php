<?php

$connection =mysqli_connect('localhost', 'root', '', 'contactt_db' );

if(isset($_POST['Send'])){
    $name = $_POST['form-name'];
    $email = $_POST['form-email'];
    $phone = $_POST['form-phone'];
    $message = $_POST['form-message'];
    $subject = $_POST['form-subject'];
   

    $request ="insert into contact_form(form-name,form-email,form-phone,form-message,from-subject) values('$name','$email','$phone','$message','$subject')";
    mysqli_query($connection, $request);
    header('location:index.php');

}else{
    echo 'Something went wrong please try again!';
}
?>