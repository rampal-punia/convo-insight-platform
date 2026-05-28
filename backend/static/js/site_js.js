// Select all check boxes on dataset_detail page for augmentation
$(document).ready(function () {
    // Close django message box
    $("#messageDialogCloseBtn").click(function () {
        $("#djangoMessage").hide();
    });

    // Fade out django message after a while
    $("#djangoMessage").fadeOut(12000);
});
